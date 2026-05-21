/**
 * Offline TensorFlow.js inference with Grad-CAM-style heatmap generation.
 */

import * as tf from '@tensorflow/tfjs';
import type { ScanResult } from '../types';
import { ICDAS_ACTIONS, DISCLAIMER } from '../types';

const IMAGE_SIZE = 224;
const MODEL_URL = '/models/model.json';

let model: tf.LayersModel | null = null;
let usingDemoMode = false;

export function isDemoMode(): boolean {
  return usingDemoMode;
}

export async function loadModel(): Promise<boolean> {
  try {
    await tf.ready();
    model = await tf.loadLayersModel(MODEL_URL);
    usingDemoMode = false;
    console.log('TF.js model loaded for offline inference');
    return true;
  } catch (e) {
    model = null;
    usingDemoMode = true;
    console.warn('Model not found — using heuristic demo mode:', e);
    return false;
  }
}

/** Preprocess image tensor from canvas/ImageData */
function preprocess(tensor: tf.Tensor3D): tf.Tensor4D {
  return tf.tidy(() => {
    let t = tf.image.resizeBilinear(tensor, [IMAGE_SIZE, IMAGE_SIZE]);
    t = t.div(255);
    // Simple normalization
    const mean = t.mean();
    const std = t.sub(mean).square().mean().sqrt().add(1e-5);
    t = t.sub(mean).div(std);
    return t.expandDims(0) as tf.Tensor4D;
  });
}

/** Demo prediction when model unavailable — for UI testing */
function demoPredict(): { grade: number; confidence: number; probs: number[] } {
  const grade = Math.floor(Math.random() * 4);
  const probs = Array(7).fill(0.05);
  probs[grade] = 0.65 + Math.random() * 0.25;
  return { grade, confidence: probs[grade] * 100, probs };
}

/** Simplified Grad-CAM using gradients w.r.t. last conv layer */
async function generateHeatmap(
  input: tf.Tensor4D,
  classIdx: number
): Promise<tf.Tensor3D> {
  if (!model) return tf.zeros([IMAGE_SIZE, IMAGE_SIZE]);

  return tf.tidy(() => {
    const prediction = model!.predict(input) as tf.Tensor;
    const loss = prediction.gather([0]).squeeze().gather([classIdx]);
    // Fallback: activation map from input gradients
    const grads = tf.grad((x: tf.Tensor) => {
      const pred = model!.predict(x.expandDims(0)) as tf.Tensor;
      return pred.gather([0]).gather([classIdx]).sum();
    });
    const grad = grads(input);
    const heatmap = grad.abs().mean(-1).squeeze();
    const norm = heatmap.div(heatmap.max().add(1e-8));
    return norm as tf.Tensor3D;
  });
}

export async function runInference(
  imageSource: HTMLImageElement | HTMLCanvasElement,
  patientId?: string
): Promise<ScanResult> {
  const start = performance.now();
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;

  if (imageSource instanceof HTMLImageElement) {
    canvas.width = imageSource.naturalWidth;
    canvas.height = imageSource.naturalHeight;
    ctx.drawImage(imageSource, 0, 0);
  } else {
    canvas.width = imageSource.width;
    canvas.height = imageSource.height;
    ctx.drawImage(imageSource, 0, 0);
  }

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const tensor = tf.browser.fromPixels(imageData) as tf.Tensor3D;
  const input = preprocess(tensor);
  tensor.dispose();

  let grade: number;
  let confidence: number;
  let probs: number[];

  if (model) {
    const pred = model.predict(input) as tf.Tensor;
    const probsData = await pred.data();
    pred.dispose();
    probs = Array.from(probsData);
    grade = probs.indexOf(Math.max(...probs));
    confidence = probs[grade] * 100;
  } else {
    const demo = demoPredict();
    grade = demo.grade;
    confidence = demo.confidence;
    probs = demo.probs;
  }

  const heatmapTensor = await generateHeatmap(input, grade);
  const overlayCanvas = await overlayHeatmap(canvas, heatmapTensor);
  heatmapTensor.dispose();
  input.dispose();

  const action = ICDAS_ACTIONS[grade] ?? ICDAS_ACTIONS[0];
  const inferenceMs = performance.now() - start;
  const isDemo = !model;

  return {
    id: crypto.randomUUID(),
    patientId,
    timestamp: new Date().toISOString(),
    icdasGrade: grade,
    confidence: Math.round(confidence * 10) / 10,
    label: action.label,
    action: action.action,
    description: action.description,
    finding: action.finding,
    recommendation: action.recommendation,
    urgency: action.urgency,
    isDemo,
    originalImage: canvas.toDataURL('image/jpeg', 0.85),
    heatmapImage: overlayCanvas.toDataURL('image/png'),
    overlayImage: overlayCanvas.toDataURL('image/png'),
    inferenceMs: Math.round(inferenceMs),
  };
}

async function overlayHeatmap(
  source: HTMLCanvasElement,
  heatmap: tf.Tensor3D
): Promise<HTMLCanvasElement> {
  const out = document.createElement('canvas');
  out.width = IMAGE_SIZE;
  out.height = IMAGE_SIZE;
  const ctx = out.getContext('2d')!;

  ctx.drawImage(source, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const heatData = await heatmap.data();
  const imgData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);

  for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
    const h = heatData[i];
    if (h > 0.3) {
      imgData.data[i * 4] = Math.min(255, imgData.data[i * 4] + h * 180);
      imgData.data[i * 4 + 1] = Math.max(0, imgData.data[i * 4 + 1] - h * 60);
      imgData.data[i * 4 + 2] = Math.max(0, imgData.data[i * 4 + 2] - h * 80);
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return out;
}

export { DISCLAIMER };
