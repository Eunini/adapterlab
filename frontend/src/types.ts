export type Dataset = {
  id: string;
  name: string;
  description: string;
  license: string;
  synthetic: boolean;
  created_at: string;
  report: {
    examples: number;
    labels: string[];
    label_counts: Record<string, number>;
    splits: Record<string, number>;
    groups: number;
    duplicates_removed: number;
    possible_pii_examples: number;
    fingerprint: string;
  };
};
export type Config = {
  strategy: "lora" | "head-only" | "full";
  rank: number;
  epochs: number;
  batch_size: number;
  max_length: number;
  learning_rate: number;
  seed: number;
};
export type Scores = {
  accuracy: number;
  macro_f1: number;
  loss: number;
  samples: number;
  labels: string[];
  confusion_matrix: number[][];
  per_class: {
    label: string;
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }[];
};
export type Result = {
  baseline: Scores;
  baseline_description: string;
  test: Scores;
  best_epoch: number;
  validation_macro_f1: number;
  trainable_parameters: number;
  total_parameters: number;
  trainable_pct: number;
  steps: number;
  elapsed_seconds: number;
  dataset_fingerprint: string;
};
export type Run = {
  id: string;
  name: string;
  dataset_id: string;
  dataset_name?: string;
  synthetic?: boolean;
  config: Config;
  status: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  step: number;
  total_steps: number;
  epoch: number;
  cancel_requested: boolean;
  result: Result | null;
  error: string | null;
  dataset_report?: Dataset["report"];
  metrics?: {
    id: number;
    step: number;
    epoch: number;
    train_loss?: number;
    validation_loss?: number;
    validation_accuracy?: number;
    validation_macro_f1?: number;
    trainable_parameters?: number;
    total_parameters?: number;
  }[];
};
export type System = {
  workers_online: number;
  queued: number;
  active: number;
  device: string;
  model_id: string;
  model_revision: string;
};
export type Prediction = {
  prediction: string;
  probabilities: { label: string; probability: number }[];
  latency_ms: number;
};
