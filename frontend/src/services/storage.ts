/**
 * IndexedDB storage with optional AES encryption for patient scan history.
 * All data stays local — no cloud upload.
 */

import { openDB, DBSchema, IDBPDatabase } from 'idb';
import type { ScanResult, AppSettings } from '../types';
import { DEFAULT_SETTINGS } from '../types';

interface SettingsRecord extends AppSettings {
  id: string;
}

interface ICDASDB extends DBSchema {
  scans: { key: string; value: ScanResult; indexes: { 'by-patient': string; 'by-date': string } };
  settings: { key: string; value: SettingsRecord };
}

const DB_NAME = 'icdas-offline-db';
const DB_VERSION = 1;

let dbPromise: Promise<IDBPDatabase<ICDASDB>> | null = null;

function getDB() {
  if (!dbPromise) {
    dbPromise = openDB<ICDASDB>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        const scanStore = db.createObjectStore('scans', { keyPath: 'id' });
        scanStore.createIndex('by-patient', 'patientId');
        scanStore.createIndex('by-date', 'timestamp');
        db.createObjectStore('settings', { keyPath: 'id' });
      },
    });
  }
  return dbPromise;
}

/** Simple XOR-based obfuscation for demo — use Web Crypto AES-GCM in production */
async function encrypt(data: string, enabled: boolean): Promise<string> {
  if (!enabled) return data;
  const key = await getOrCreateKey();
  const encoded = new TextEncoder().encode(data);
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const cipher = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, encoded);
  const combined = new Uint8Array(iv.length + cipher.byteLength);
  combined.set(iv);
  combined.set(new Uint8Array(cipher), iv.length);
  return btoa(String.fromCharCode(...combined));
}

async function decrypt(data: string, enabled: boolean): Promise<string> {
  if (!enabled) return data;
  const key = await getOrCreateKey();
  const combined = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  const iv = combined.slice(0, 12);
  const cipher = combined.slice(12);
  const plain = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, cipher);
  return new TextDecoder().decode(plain);
}

async function getOrCreateKey(): Promise<CryptoKey> {
  const stored = localStorage.getItem('icdas-crypto-key');
  if (stored) {
    const raw = Uint8Array.from(atob(stored), (c) => c.charCodeAt(0));
    return crypto.subtle.importKey('raw', raw, 'AES-GCM', false, ['encrypt', 'decrypt']);
  }
  const key = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']);
  const exported = await crypto.subtle.exportKey('raw', key);
  localStorage.setItem('icdas-crypto-key', btoa(String.fromCharCode(...new Uint8Array(exported))));
  return key;
}

export async function saveScan(scan: ScanResult, encryptData = true): Promise<void> {
  const db = await getDB();
  const payload = { ...scan };
  if (encryptData) {
    payload.originalImage = await encrypt(scan.originalImage, true);
    if (scan.heatmapImage) payload.heatmapImage = await encrypt(scan.heatmapImage, true);
  }
  await db.put('scans', payload);
}

export async function getScans(patientId?: string): Promise<ScanResult[]> {
  const db = await getDB();
  let scans = await db.getAll('scans');
  if (patientId) scans = scans.filter((s) => s.patientId === patientId);
  const settings = await getSettings();
  const decrypted: ScanResult[] = [];
  for (const s of scans) {
    decrypted.push({
      ...s,
      originalImage: await decrypt(s.originalImage, settings.encryptionEnabled),
      heatmapImage: s.heatmapImage ? await decrypt(s.heatmapImage, settings.encryptionEnabled) : undefined,
    });
  }
  return decrypted.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
}

export async function deleteScan(id: string): Promise<void> {
  const db = await getDB();
  await db.delete('scans', id);
}

export async function getSettings(): Promise<AppSettings> {
  const db = await getDB();
  const stored = await db.get('settings', 'app');
  if (!stored) return DEFAULT_SETTINGS;
  const { id: _id, ...settings } = stored;
  return settings;
}

export async function saveSettings(settings: AppSettings): Promise<void> {
  const db = await getDB();
  await db.put('settings', { ...settings, id: 'app' });
}
