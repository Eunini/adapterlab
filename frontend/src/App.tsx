import { useCallback, useEffect, useRef, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowRight,
  ArrowSquareOut,
  ArrowsClockwise,
  BezierCurve,
  CaretRight,
  ChartLine,
  Check,
  CheckCircle,
  CircleNotch,
  Clock,
  Cpu,
  Database,
  DownloadSimple,
  Flask,
  GitBranch,
  GithubLogo,
  Plus,
  RocketLaunch,
  Stack,
  Stop,
  UploadSimple,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import type { Config, Dataset, Prediction, Run, System } from "./types";

const terminal = ["succeeded", "failed", "cancelled"];
const pct = (value: number | undefined) =>
  value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
const number = (value: number | undefined) =>
  value === undefined
    ? "—"
    : new Intl.NumberFormat("en", {
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(value);
const elapsed = (seconds: number | undefined) =>
  seconds === undefined
    ? "—"
    : seconds < 60
      ? `${seconds.toFixed(1)}s`
      : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
const initialConfig: Config = {
  strategy: "lora",
  rank: 8,
  epochs: 4,
  batch_size: 16,
  max_length: 64,
  learning_rate: 0.001,
  seed: 42,
};

export default function App() {
  const [page, setPage] = useState("Experiments");
  const [detailTab, setDetailTab] = useState("Evaluation");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [system, setSystem] = useState<System>();
  const [selectedId, setSelectedId] = useState("");
  const [selected, setSelected] = useState<Run>();
  const [compare, setCompare] = useState<string[]>([]);
  const [showRun, setShowRun] = useState(false);
  const [showDataset, setShowDataset] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [notice, setNotice] = useState("");
  const [runName, setRunName] = useState("support-lora");
  const [datasetId, setDatasetId] = useState("");
  const [config, setConfig] = useState<Config>(initialConfig);
  const [datasetName, setDatasetName] = useState("");
  const [datasetLicense, setDatasetLicense] = useState("User-provided data");
  const [datasetFile, setDatasetFile] = useState<File>();
  const [permission, setPermission] = useState(false);
  const [synthetic, setSynthetic] = useState(false);
  const [predictionText, setPredictionText] = useState(
    "I cannot sign in after resetting my password.",
  );
  const [prediction, setPrediction] = useState<Prediction>();
  const [token, setToken] = useState(
    () => sessionStorage.getItem("adapterlab-operator") || "",
  );
  const [tokenInput, setTokenInput] = useState("");
  const [authRequired, setAuthRequired] = useState(false);

  const request = useCallback(
    async (path: string, options: RequestInit = {}) => {
      const response = await fetch(`/api${path}`, {
        ...options,
        headers: {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...options.headers,
        },
      });
      if (response.status === 401) {
        setAuthRequired(true);
        throw new Error("Operator access is required.");
      }
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(
          typeof body.detail === "string"
            ? body.detail
            : body.detail
                ?.map((item: { msg: string }) => item.msg)
                .join("; ") || `Request failed (${response.status}).`,
        );
      }
      setAuthRequired(false);
      return response;
    },
    [token],
  );

  const refresh = useCallback(async () => {
    try {
      const [datasets, runs, system] = await Promise.all(
        ["/datasets", "/runs", "/system"].map(async (path) =>
          (await request(path)).json(),
        ),
      );
      setDatasets(datasets);
      setRuns(runs);
      setSystem(system);
      setDatasetId((current) => current || datasets[0]?.id || "");
      setSelectedId((current) =>
        runs.some((run: Run) => run.id === current)
          ? current
          : runs[0]?.id || "",
      );
      setConnectionError("");
    } catch (cause) {
      setConnectionError(
        cause instanceof Error
          ? cause.message
          : "Unable to reach the training API.",
      );
    }
  }, [request]);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 2500);
    return () => window.clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    let active = true;
    const refreshRun = async () => {
      if (!selectedId) {
        setSelected(undefined);
        return;
      }
      try {
        const row = await (await request(`/runs/${selectedId}`)).json();
        if (active) setSelected(row);
      } catch (cause) {
        if (active)
          setError(
            cause instanceof Error ? cause.message : "Unable to load run.",
          );
      }
    };
    setPrediction(undefined);
    setSelected(undefined);
    void refreshRun();
    const interval = window.setInterval(() => void refreshRun(), 2000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [selectedId, request]);

  async function action(work: () => Promise<void>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await work();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Operation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function submitRun() {
    const row = await (
      await request("/runs", {
        method: "POST",
        body: JSON.stringify({ name: runName, dataset_id: datasetId, config }),
      })
    ).json();
    setSelectedId(row.id);
    setShowRun(false);
    setPage("Experiments");
    await refresh();
    setNotice(
      "Run queued. The worker will load the pinned base model and begin training.",
    );
  }

  async function importDataset() {
    if (!datasetFile || !permission) return;
    if (datasetFile.size > 8_000_000)
      throw new Error("Choose a dataset smaller than 8 MB.");
    const content = await datasetFile.text();
    const records = content.trim().startsWith("[")
      ? JSON.parse(content)
      : content
          .split(/\r?\n/)
          .filter((line) => line.trim())
          .map((line) => JSON.parse(line));
    const dataset = await (
      await request("/datasets", {
        method: "POST",
        body: JSON.stringify({
          name: datasetName,
          license: datasetLicense,
          synthetic,
          records,
        }),
      })
    ).json();
    setShowDataset(false);
    setDatasetId(dataset.id);
    await refresh();
    setNotice(
      `Registered ${dataset.report.examples} examples with group-separated splits.`,
    );
  }

  async function download(run: Run) {
    const response = await request(`/runs/${run.id}/export`);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = `adapterlab-${run.id.slice(0, 8)}.zip`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice(
      "Model bundle downloaded with evaluation results and file checksums.",
    );
  }

  const completed = runs.filter((run) => run.status === "succeeded");
  const best = completed.reduce<number | undefined>(
    (value, run) => Math.max(value || 0, run.result?.test.macro_f1 || 0),
    undefined,
  );
  const chosenDataset = datasets.find((dataset) => dataset.id === datasetId);
  const comparison = runs.filter((run) => compare.includes(run.id));
  const result = selected?.result;
  const grouped = new Map<
    number,
    { step: number; train?: number; validation?: number }
  >();
  selected?.metrics?.forEach((point) => {
    const entry = grouped.get(point.step) || { step: point.step };
    if (point.train_loss !== undefined) entry.train = point.train_loss;
    if (point.validation_loss !== undefined)
      entry.validation = point.validation_loss;
    grouped.set(point.step, entry);
  });
  const curve = [...grouped.values()].sort(
    (left, right) => left.step - right.step,
  );
  const params = selected?.metrics?.find((point) => point.total_parameters);
  const trainable =
    result?.trainable_parameters ?? params?.trainable_parameters;
  const total = result?.total_parameters ?? params?.total_parameters;

  return (
    <div className="app">
      <header>
        <a className="brand" href="/">
          <span className="brand-mark">
            <BezierCurve weight="bold" size={23} />
          </span>
          adapter<span>lab</span>
        </a>
        <nav aria-label="Workspace navigation">
          {["Experiments", "Datasets", "Model registry"].map((name) => (
            <button
              key={name}
              aria-pressed={page === name}
              className={page === name ? "active" : ""}
              onClick={() => setPage(name)}
            >
              {name}
            </button>
          ))}
        </nav>
        <div className="header-right">
          <span
            className={`worker-state ${system?.workers_online ? "online" : ""}`}
          >
            <span className="dot" />
            {system?.workers_online || 0} worker
            {system?.workers_online === 1 ? "" : "s"}
            <span className="separator">/</span>CPU
          </span>
          <a
            href="https://github.com/Eunini/adapterlab"
            aria-label="Source on GitHub"
            target="_blank"
            rel="noreferrer"
          >
            <GithubLogo size={23} />
          </a>
        </div>
      </header>
      <main>
        <div className="breadcrumb">
          <span className="workspace-icon">A</span>Local workspace
          <CaretRight size={12} />
          <strong>{page}</strong>
        </div>
        <div className="page-title">
          <div>
            <h1>
              {page === "Experiments"
                ? "Training workspace"
                : page === "Datasets"
                  ? "Dataset versions"
                  : "Trained models"}
            </h1>
            <p>
              {page === "Experiments"
                ? "Adapt a pretrained model. Compare results on held-out data."
                : page === "Datasets"
                  ? "Validate examples and keep related groups in the same split."
                  : "Download trained weights, evaluation results, and reproducible configurations."}
            </p>
          </div>
          <button
            className="button primary"
            onClick={() =>
              page === "Datasets" ? setShowDataset(true) : setShowRun(true)
            }
          >
            {page === "Datasets" ? (
              <UploadSimple size={17} />
            ) : (
              <Plus size={18} />
            )}
            {page === "Datasets" ? "Import dataset" : "New experiment"}
          </button>
        </div>

        {(error || connectionError) && (
          <div className="banner error" role="alert">
            <WarningCircle size={18} />
            <span>{error || connectionError}</span>
            <button
              aria-label="Dismiss error"
              onClick={() => {
                setError("");
                setConnectionError("");
              }}
            >
              <X size={15} />
            </button>
          </div>
        )}
        {notice && (
          <div className="banner success" role="status">
            <CheckCircle size={18} />
            <span>{notice}</span>
            <button aria-label="Dismiss notice" onClick={() => setNotice("")}>
              <X size={15} />
            </button>
          </div>
        )}
        {authRequired && (
          <form
            className="access panel"
            onSubmit={(event) => {
              event.preventDefault();
              sessionStorage.setItem("adapterlab-operator", tokenInput);
              setToken(tokenInput);
              setTokenInput("");
            }}
          >
            <h2>Operator access</h2>
            <label>
              Workspace token
              <input
                type="password"
                value={tokenInput}
                autoComplete="off"
                onChange={(event) => setTokenInput(event.target.value)}
                required
              />
            </label>
            <button className="button primary">Connect workspace</button>
          </form>
        )}

        {!authRequired && (
          <>
            <section className="summary-strip" aria-label="Workspace metrics">
              <Summary
                icon={<Database size={17} />}
                label="Dataset versions"
                value={String(datasets.length)}
                sub="Validated & fingerprinted"
              />
              <Summary
                icon={<CheckCircle size={17} />}
                label="Completed runs"
                value={String(completed.length)}
                sub={`${system?.queued || 0} queued · ${system?.active || 0} active`}
              />
              <Summary
                icon={<ChartLine size={17} />}
                label="Best test macro F1"
                value={pct(best)}
                sub="Across completed experiments"
              />
              <Summary
                icon={<GitBranch size={17} />}
                label="Selected trainable parameters"
                value={
                  total && trainable !== undefined
                    ? `${((100 * trainable) / total).toFixed(2)}%`
                    : "—"
                }
                sub={
                  total
                    ? `${number(trainable)} of ${number(total)} parameters`
                    : "Available after model preparation"
                }
              />
            </section>

            {page === "Experiments" && (
              <>
                <section className="panel experiment-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>
                        Experiments <span className="count">{runs.length}</span>
                      </h2>
                      <p>
                        Select a run to inspect its loss curve, evaluations, and
                        model files.
                      </p>
                    </div>
                    <button
                      className="icon-button"
                      aria-label="Refresh workspace"
                      onClick={() => void refresh()}
                    >
                      <ArrowsClockwise size={17} />
                    </button>
                  </div>
                  {runs.length ? (
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th className="checkbox-cell">
                              <span className="sr-only">Compare</span>
                            </th>
                            <th>Experiment</th>
                            <th>Recipe</th>
                            <th>Status</th>
                            <th>Test F1</th>
                            <th>Duration</th>
                            <th />
                          </tr>
                        </thead>
                        <tbody>
                          {runs.map((run) => (
                            <tr
                              key={run.id}
                              className={
                                selectedId === run.id ? "selected-row" : ""
                              }
                            >
                              <td>
                                <input
                                  type="checkbox"
                                  aria-label={`Compare ${run.name}`}
                                  checked={compare.includes(run.id)}
                                  disabled={run.status !== "succeeded"}
                                  onChange={(event) => {
                                    if (
                                      event.target.checked &&
                                      comparison.some(
                                        (item) =>
                                          item.dataset_id !== run.dataset_id,
                                      )
                                    ) {
                                      setError(
                                        "Compare runs on the same dataset version.",
                                      );
                                      return;
                                    }
                                    setCompare(
                                      event.target.checked
                                        ? [...compare, run.id]
                                        : compare.filter((id) => id !== run.id),
                                    );
                                  }}
                                />
                              </td>
                              <td>
                                <button
                                  className="run-link"
                                  onClick={() => setSelectedId(run.id)}
                                >
                                  <strong>{run.name}</strong>
                                  <span>
                                    {run.dataset_name}{" "}
                                    <span className="mono">
                                      · {run.id.slice(0, 8)}
                                    </span>
                                  </span>
                                </button>
                              </td>
                              <td>
                                <span className="recipe">
                                  {run.config.strategy === "lora"
                                    ? `LoRA · r${run.config.rank}`
                                    : run.config.strategy}
                                </span>
                              </td>
                              <td>
                                <Status value={run.status} />
                              </td>
                              <td className="mono score">
                                {pct(run.result?.test.macro_f1)}
                              </td>
                              <td className="mono subtle">
                                {elapsed(run.result?.elapsed_seconds)}
                              </td>
                              <td>
                                <button
                                  className="icon-button"
                                  aria-label={`Inspect ${run.name}`}
                                  onClick={() => setSelectedId(run.id)}
                                >
                                  <ArrowRight size={16} />
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="empty">
                      <Flask size={32} />
                      <h3>No experiments yet</h3>
                      <p>
                        Start with the bundled synthetic support-intent dataset
                        or import your own.
                      </p>
                      <button
                        className="button"
                        onClick={() => setShowRun(true)}
                      >
                        Configure a training run <ArrowRight size={15} />
                      </button>
                    </div>
                  )}
                </section>

                {comparison.length >= 2 && (
                  <section className="panel">
                    <div className="panel-heading">
                      <div>
                        <h2>Run comparison</h2>
                        <p>Same dataset fingerprint and split assignments.</p>
                      </div>
                      <button
                        className="text-button"
                        onClick={() => setCompare([])}
                      >
                        Clear selection
                      </button>
                    </div>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Run</th>
                            <th>Test accuracy</th>
                            <th>Test macro F1</th>
                            <th>Trainable</th>
                            <th>Time</th>
                          </tr>
                        </thead>
                        <tbody>
                          {comparison.map((run) => (
                            <tr key={run.id}>
                              <td>{run.name}</td>
                              <td className="mono">
                                {pct(run.result?.test.accuracy)}
                              </td>
                              <td className="mono">
                                {pct(run.result?.test.macro_f1)}
                              </td>
                              <td className="mono">
                                {run.result?.trainable_pct}%
                              </td>
                              <td>{elapsed(run.result?.elapsed_seconds)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                )}

                {selected && (
                  <>
                    <div className="run-section-title">
                      <div>
                        <Flask size={19} />
                        <h2>{selected.name}</h2>
                        <Status value={selected.status} />
                        {selected.synthetic && (
                          <span className="badge amber">Synthetic dataset</span>
                        )}
                      </div>
                      <div>
                        {!terminal.includes(selected.status) && (
                          <button
                            className="button danger"
                            disabled={busy || selected.status === "cancelling"}
                            onClick={() =>
                              void action(async () => {
                                await request(`/runs/${selected.id}/cancel`, {
                                  method: "POST",
                                });
                                await refresh();
                              })
                            }
                          >
                            <Stop size={15} />
                            Cancel run
                          </button>
                        )}
                        <button
                          className="button"
                          onClick={() => {
                            setConfig(selected.config);
                            setDatasetId(selected.dataset_id);
                            setRunName(`${selected.name}-copy`);
                            setShowRun(true);
                          }}
                        >
                          Use configuration
                        </button>
                      </div>
                    </div>
                    {!terminal.includes(selected.status) && (
                      <div className="training-progress">
                        <div>
                          <CircleNotch className="spin" size={17} />
                          <strong>
                            {selected.status === "queued"
                              ? "Waiting for a worker"
                              : selected.status === "preparing"
                                ? "Preparing tokenizer and pretrained weights"
                                : selected.status === "evaluating"
                                  ? "Evaluating the selected checkpoint"
                                  : selected.status === "cancelling"
                                    ? "Stopping after the current batch"
                                    : `Training · epoch ${selected.epoch.toFixed(1)} of ${selected.config.epochs}`}
                          </strong>
                          <span className="mono">
                            {selected.step} / {selected.total_steps || "—"}{" "}
                            steps
                          </span>
                        </div>
                        <progress
                          max={selected.total_steps || 1}
                          value={selected.step}
                        />
                      </div>
                    )}
                    {selected.error && (
                      <div className="banner error">
                        <WarningCircle size={18} />
                        {selected.error}
                      </div>
                    )}
                    <div className="run-layout">
                      <section className="panel loss-panel">
                        <div className="panel-heading">
                          <div>
                            <h2>Loss over training</h2>
                            <p>
                              Cross-entropy measured by the training worker.
                            </p>
                          </div>
                          <div className="legend">
                            <span>
                              <i />
                              Training
                            </span>
                            <span>
                              <i className="validation" />
                              Validation
                            </span>
                          </div>
                        </div>
                        {curve.some((point) => point.train !== undefined) ? (
                          <div
                            className="loss-chart"
                            role="img"
                            aria-label="Training and validation loss by optimizer step"
                          >
                            <ResponsiveContainer width="100%" height="100%">
                              <ComposedChart
                                data={curve}
                                margin={{
                                  top: 12,
                                  right: 18,
                                  bottom: 4,
                                  left: -15,
                                }}
                              >
                                <defs>
                                  <linearGradient
                                    id="loss-fill"
                                    x1="0"
                                    y1="0"
                                    x2="0"
                                    y2="1"
                                  >
                                    <stop
                                      offset="0%"
                                      stopColor="#3870dd"
                                      stopOpacity={0.12}
                                    />
                                    <stop
                                      offset="100%"
                                      stopColor="#3870dd"
                                      stopOpacity={0}
                                    />
                                  </linearGradient>
                                </defs>
                                <CartesianGrid
                                  vertical={false}
                                  stroke="#edf0f5"
                                  strokeDasharray="3 4"
                                />
                                <XAxis
                                  dataKey="step"
                                  tickLine={false}
                                  axisLine={false}
                                  minTickGap={35}
                                  tick={{ fontSize: 10, fill: "#9299a7" }}
                                />
                                <YAxis
                                  tickLine={false}
                                  axisLine={false}
                                  tick={{ fontSize: 10, fill: "#9299a7" }}
                                />
                                <Tooltip
                                  formatter={(value, name) => [
                                    Number(value).toFixed(4),
                                    name,
                                  ]}
                                  labelFormatter={(value) => `Step ${value}`}
                                />
                                <Area
                                  type="monotone"
                                  dataKey="train"
                                  name="Training"
                                  stroke="#3269d6"
                                  strokeWidth={2}
                                  fill="url(#loss-fill)"
                                  isAnimationActive={false}
                                  connectNulls
                                />
                                <Line
                                  dataKey="validation"
                                  name="Validation"
                                  stroke="#e59b50"
                                  strokeWidth={2}
                                  strokeDasharray="5 4"
                                  dot={{ r: 3 }}
                                  isAnimationActive={false}
                                  connectNulls
                                />
                              </ComposedChart>
                            </ResponsiveContainer>
                          </div>
                        ) : (
                          <div className="empty chart-empty">
                            <ChartLine size={30} />
                            <p>
                              Training measurements will appear after the first
                              batch.
                            </p>
                          </div>
                        )}
                        <div className="panel-footer">
                          <span>Optimizer step</span>
                          <span>
                            Best checkpoint selected by validation macro F1
                            {result ? ` · epoch ${result.best_epoch}` : ""}
                          </span>
                        </div>
                      </section>

                      <section className="panel configuration">
                        <div className="panel-heading">
                          <h2>Run configuration</h2>
                          <Cpu size={18} />
                        </div>
                        <dl>
                          <div>
                            <dt>Base model</dt>
                            <dd>
                              BERT Tiny{" "}
                              <span className="subtle">2L / 128H</span>
                            </dd>
                          </div>
                          <div>
                            <dt>Method</dt>
                            <dd>
                              {selected.config.strategy === "lora"
                                ? `LoRA · rank ${selected.config.rank}`
                                : selected.config.strategy}
                            </dd>
                          </div>
                          <div>
                            <dt>Learning rate</dt>
                            <dd className="mono">
                              {selected.config.learning_rate}
                            </dd>
                          </div>
                          <div>
                            <dt>Batch / sequence</dt>
                            <dd className="mono">
                              {selected.config.batch_size} /{" "}
                              {selected.config.max_length}
                            </dd>
                          </div>
                          <div>
                            <dt>Epochs / seed</dt>
                            <dd className="mono">
                              {selected.config.epochs} / {selected.config.seed}
                            </dd>
                          </div>
                          <div>
                            <dt>Trainable weights</dt>
                            <dd className="mono">
                              {number(trainable)}{" "}
                              <span className="subtle">/ {number(total)}</span>
                            </dd>
                          </div>
                        </dl>
                        <div className="fingerprint">
                          <span>DATASET FINGERPRINT</span>
                          <code>{selected.dataset_id.slice(0, 24)}…</code>
                        </div>
                      </section>
                    </div>

                    <section className="panel detail-panel">
                      <div className="detail-tabs" aria-label="Run details">
                        {["Evaluation", "Inference", "Artifacts"].map((tab) => (
                          <button
                            key={tab}
                            className={detailTab === tab ? "active" : ""}
                            aria-pressed={detailTab === tab}
                            onClick={() => setDetailTab(tab)}
                          >
                            {tab === "Evaluation" ? (
                              <ChartLine size={16} />
                            ) : tab === "Inference" ? (
                              <RocketLaunch size={16} />
                            ) : (
                              <Stack size={16} />
                            )}
                            {tab}
                          </button>
                        ))}
                      </div>
                      {detailTab === "Evaluation" &&
                        (result ? (
                          <div className="evaluation-content">
                            <div className="evaluation-top">
                              <div>
                                <span className="field-label">
                                  HELD-OUT TEST SPLIT
                                </span>
                                <h3>Measured after checkpoint selection</h3>
                                <p>
                                  {result.test.samples} examples · test data
                                  never enters the optimizer
                                </p>
                              </div>
                              <div className="evaluation-scores">
                                <div>
                                  <span>Accuracy</span>
                                  <strong>{pct(result.test.accuracy)}</strong>
                                </div>
                                <div>
                                  <span>Macro F1</span>
                                  <strong>{pct(result.test.macro_f1)}</strong>
                                </div>
                                <div>
                                  <span>Unadapted head F1</span>
                                  <strong className="subtle">
                                    {pct(result.baseline.macro_f1)}
                                  </strong>
                                </div>
                              </div>
                            </div>
                            <div className="evaluation-grid">
                              <div>
                                <h4>Confusion matrix</h4>
                                <p className="matrix-caption">
                                  Rows: actual · columns: predicted
                                </p>
                                <div className="table-scroll">
                                  <table className="matrix">
                                    <thead>
                                      <tr>
                                        <th />
                                        {result.test.labels.map((label) => (
                                          <th key={label}>
                                            {label.slice(0, 5)}
                                          </th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {result.test.confusion_matrix.map(
                                        (row, index) => (
                                          <tr key={result.test.labels[index]}>
                                            <th>{result.test.labels[index]}</th>
                                            {row.map((count, col) => (
                                              <td
                                                key={col}
                                                style={{
                                                  background: `rgba(47, 102, 216, ${count ? 0.08 + (0.5 * count) / Math.max(1, ...result.test.confusion_matrix.flat()) : 0.025})`,
                                                }}
                                              >
                                                {count}
                                              </td>
                                            ))}
                                          </tr>
                                        ),
                                      )}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                              <div>
                                <h4>Per-label performance</h4>
                                <table className="per-class">
                                  <thead>
                                    <tr>
                                      <th>Label</th>
                                      <th>F1</th>
                                      <th>Support</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {result.test.per_class.map((row) => (
                                      <tr key={row.label}>
                                        <td>{row.label}</td>
                                        <td className="mono">{pct(row.f1)}</td>
                                        <td className="mono subtle">
                                          {row.support}
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                            <p className="evaluation-note">
                              The unadapted baseline uses a pretrained encoder
                              with a newly initialized classification head.{" "}
                              {selected.synthetic
                                ? "These results describe synthetic examples, not production support accuracy."
                                : "Validate performance on representative deployment data before use."}
                            </p>
                          </div>
                        ) : (
                          <div className="empty">
                            <ChartLine size={28} />
                            <h3>Evaluation is not available yet</h3>
                            <p>
                              A completed run adds held-out scores and a
                              confusion matrix here.
                            </p>
                          </div>
                        ))}
                      {detailTab === "Inference" && (
                        <div className="inference-content">
                          <div>
                            <h3>Test the trained classifier</h3>
                            <p>
                              Predictions run locally using this experiment’s
                              selected checkpoint.
                            </p>
                            <label htmlFor="prediction-text">
                              Support request
                            </label>
                            <textarea
                              id="prediction-text"
                              value={predictionText}
                              onChange={(event) =>
                                setPredictionText(event.target.value)
                              }
                              maxLength={3000}
                            />
                            <button
                              className="button primary"
                              disabled={
                                busy ||
                                selected.status !== "succeeded" ||
                                predictionText.trim().length < 3
                              }
                              onClick={() =>
                                void action(async () => {
                                  const response = await request(
                                    `/runs/${selected.id}/predict`,
                                    {
                                      method: "POST",
                                      body: JSON.stringify({
                                        text: predictionText,
                                      }),
                                    },
                                  );
                                  setPrediction(await response.json());
                                })
                              }
                            >
                              <RocketLaunch size={16} />
                              Run inference
                            </button>
                          </div>
                          <div className="prediction-results">
                            {prediction ? (
                              <>
                                <div className="prediction-title">
                                  <span>Predicted intent</span>
                                  <strong>{prediction.prediction}</strong>
                                  <small>
                                    {prediction.latency_ms} ms forward pass
                                  </small>
                                </div>
                                {prediction.probabilities.map((row) => (
                                  <div className="probability" key={row.label}>
                                    <div>
                                      <span>{row.label}</span>
                                      <strong className="mono">
                                        {pct(row.probability)}
                                      </strong>
                                    </div>
                                    <div className="probability-track">
                                      <span
                                        style={{
                                          width: `${row.probability * 100}%`,
                                        }}
                                      />
                                    </div>
                                  </div>
                                ))}
                                <p>
                                  Softmax scores are not calibrated
                                  probabilities.
                                </p>
                              </>
                            ) : (
                              <div className="prediction-empty">
                                <RocketLaunch size={30} />
                                <p>
                                  {selected.status === "succeeded"
                                    ? "Run an example to inspect model scores."
                                    : "Complete training to enable inference."}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                      {detailTab === "Artifacts" && (
                        <div className="artifacts-content">
                          <div>
                            <span className="artifact-icon">
                              <Stack size={27} />
                            </span>
                            <h3>Portable model bundle</h3>
                            <p>
                              Weights, tokenizer, label mapping, dataset
                              fingerprint, evaluation, training history, and
                              SHA-256 checksums.
                            </p>
                            <button
                              className="button primary"
                              disabled={busy || selected.status !== "succeeded"}
                              onClick={() =>
                                void action(() => download(selected))
                              }
                            >
                              <DownloadSimple size={16} />
                              Download model bundle
                            </button>
                          </div>
                          <ul>
                            {[
                              "checkpoint/ · safetensors weights and tokenizer",
                              "run.json · recipe and pinned base revision",
                              "evaluation.json · baseline and held-out scores",
                              "dataset_manifest.json · splits and fingerprint",
                              "checksums.json · artifact integrity",
                            ].map((file) => (
                              <li key={file}>
                                <Check size={14} />
                                <code>{file}</code>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </section>
                  </>
                )}
              </>
            )}

            {page === "Datasets" && (
              <div className="dataset-grid">
                {datasets.map((dataset) => (
                  <section className="panel dataset-card" key={dataset.id}>
                    <div className="dataset-card-title">
                      <span className="dataset-icon">
                        <Database size={24} />
                      </span>
                      <span
                        className={`badge ${dataset.synthetic ? "amber" : "gray"}`}
                      >
                        {dataset.synthetic ? "Synthetic" : "User-provided"}
                      </span>
                    </div>
                    <h2>{dataset.name}</h2>
                    <p>{dataset.description || dataset.license}</p>
                    <code className="dataset-hash">
                      {dataset.id.slice(0, 20)}
                    </code>
                    <div className="dataset-stats">
                      <div>
                        <strong>{dataset.report.examples}</strong>
                        <span>examples</span>
                      </div>
                      <div>
                        <strong>{dataset.report.labels.length}</strong>
                        <span>labels</span>
                      </div>
                      <div>
                        <strong>{dataset.report.groups}</strong>
                        <span>groups</span>
                      </div>
                    </div>
                    <div className="split-bar" aria-label="Dataset splits">
                      {["train", "validation", "test"].map((split) => (
                        <span
                          key={split}
                          className={split}
                          style={{ flex: dataset.report.splits[split] }}
                        />
                      ))}
                    </div>
                    <div className="split-labels">
                      {["train", "validation", "test"].map((split) => (
                        <span key={split}>
                          {split}{" "}
                          <strong>{dataset.report.splits[split]}</strong>
                        </span>
                      ))}
                    </div>
                    <div className="dataset-checks">
                      <span>
                        <CheckCircle size={14} />
                        Groups separated across splits
                      </span>
                      <span>
                        <CheckCircle size={14} />
                        {dataset.report.duplicates_removed} duplicates removed
                      </span>
                      {dataset.report.possible_pii_examples > 0 && (
                        <span className="pii-warning">
                          <WarningCircle size={14} />
                          Review {dataset.report.possible_pii_examples} examples
                          for personal data
                        </span>
                      )}
                    </div>
                    <button
                      className="button"
                      onClick={() => {
                        setDatasetId(dataset.id);
                        setShowRun(true);
                      }}
                    >
                      Train on this version <ArrowRight size={15} />
                    </button>
                  </section>
                ))}
              </div>
            )}

            {page === "Model registry" && (
              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Completed checkpoints</h2>
                    <p>Only successful training runs appear in the registry.</p>
                  </div>
                  <span className="count">{completed.length}</span>
                </div>
                {completed.length ? (
                  <div className="registry-list">
                    {completed.map((run) => (
                      <article key={run.id}>
                        <span className="registry-icon">
                          <GitBranch size={23} />
                        </span>
                        <div>
                          <h3>{run.name}</h3>
                          <p>
                            {run.config.strategy} · {run.dataset_name} ·{" "}
                            <span className="mono">{run.id.slice(0, 8)}</span>
                          </p>
                        </div>
                        <div className="registry-score">
                          <strong>{pct(run.result?.test.macro_f1)}</strong>
                          <span>test macro F1</span>
                        </div>
                        <button
                          className="button"
                          onClick={() => {
                            setSelectedId(run.id);
                            setDetailTab("Inference");
                            setPage("Experiments");
                          }}
                        >
                          Try model <ArrowRight size={15} />
                        </button>
                        <button
                          className="icon-button"
                          disabled={busy}
                          aria-label={`Download ${run.name}`}
                          onClick={() => void action(() => download(run))}
                        >
                          <DownloadSimple size={18} />
                        </button>
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className="empty">
                    <Stack size={32} />
                    <h3>No trained checkpoints</h3>
                    <p>Complete an experiment to export and test its model.</p>
                  </div>
                )}
              </section>
            )}

            <footer>
              <span>AdapterLab · Local training workspace</span>
              <span>
                Base model: Google BERT Tiny · Apache 2.0
                <a
                  href="https://huggingface.co/google/bert_uncased_L-2_H-128_A-2"
                  target="_blank"
                  rel="noreferrer"
                  aria-label="Base model card"
                >
                  <ArrowSquareOut size={13} />
                </a>
              </span>
            </footer>
          </>
        )}
      </main>

      {showRun && (
        <Modal
          title="New experiment"
          error={error}
          onClose={() => setShowRun(false)}
        >
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void action(submitRun);
            }}
          >
            <p className="modal-description">
              Train a classifier on a validated dataset version. The base
              encoder is pinned to an immutable revision.
            </p>
            <label>
              Experiment name
              <input
                required
                minLength={2}
                maxLength={80}
                value={runName}
                onChange={(event) => setRunName(event.target.value)}
              />
            </label>
            <label>
              Dataset version
              <select
                value={datasetId}
                onChange={(event) => setDatasetId(event.target.value)}
                required
              >
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} · {dataset.report.examples} examples
                  </option>
                ))}
              </select>
            </label>
            {chosenDataset && (
              <div className="dataset-summary">
                <Database size={17} />
                {chosenDataset.report.labels.length} labels ·{" "}
                {chosenDataset.report.splits.train} train /{" "}
                {chosenDataset.report.splits.validation} validation /{" "}
                {chosenDataset.report.splits.test} test
                {chosenDataset.synthetic && (
                  <span className="badge amber">Synthetic</span>
                )}
              </div>
            )}
            <div className="form-grid">
              <label>
                Training recipe
                <select
                  value={config.strategy}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      strategy: event.target.value as Config["strategy"],
                    })
                  }
                >
                  <option value="lora">LoRA adapter + classifier</option>
                  <option value="head-only">Classifier head only</option>
                  <option value="full">Full fine-tuning</option>
                </select>
              </label>
              <label>
                LoRA rank
                <select
                  disabled={config.strategy !== "lora"}
                  value={config.rank}
                  onChange={(event) =>
                    setConfig({ ...config, rank: Number(event.target.value) })
                  }
                >
                  {[4, 8, 16].map((rank) => (
                    <option key={rank} value={rank}>
                      {rank}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Epochs
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={config.epochs}
                  onChange={(event) =>
                    setConfig({ ...config, epochs: Number(event.target.value) })
                  }
                  required
                />
              </label>
              <label>
                Learning rate
                <input
                  type="number"
                  min={0.00001}
                  max={0.01}
                  step={0.00001}
                  value={config.learning_rate}
                  onChange={(event) =>
                    setConfig({
                      ...config,
                      learning_rate: Number(event.target.value),
                    })
                  }
                  required
                />
              </label>
            </div>
            <details>
              <summary>Batch size, sequence length, and seed</summary>
              <div className="form-grid advanced">
                <label>
                  Batch size
                  <select
                    value={config.batch_size}
                    onChange={(event) =>
                      setConfig({
                        ...config,
                        batch_size: Number(event.target.value),
                      })
                    }
                  >
                    {[4, 8, 16, 32].map((size) => (
                      <option key={size} value={size}>
                        {size}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Sequence length
                  <input
                    type="number"
                    min={16}
                    max={192}
                    value={config.max_length}
                    onChange={(event) =>
                      setConfig({
                        ...config,
                        max_length: Number(event.target.value),
                      })
                    }
                  />
                </label>
                <label>
                  Random seed
                  <input
                    type="number"
                    min={0}
                    value={config.seed}
                    onChange={(event) =>
                      setConfig({ ...config, seed: Number(event.target.value) })
                    }
                  />
                </label>
              </div>
            </details>
            <div className="modal-footer">
              <span>
                <Cpu size={15} />
                Local CPU worker
              </span>
              <button className="button primary" disabled={busy || !datasetId}>
                {busy ? (
                  <CircleNotch className="spin" size={16} />
                ) : (
                  <Flask size={16} />
                )}
                Queue experiment
              </button>
            </div>
          </form>
        </Modal>
      )}

      {showDataset && (
        <Modal
          title="Import a dataset"
          error={error}
          onClose={() => setShowDataset(false)}
        >
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void action(importDataset);
            }}
          >
            <p className="modal-description">
              Upload JSONL or a JSON array with <code>text</code>,{" "}
              <code>label</code>, and optional <code>group</code> fields. Each
              label needs at least five distinct groups.
            </p>
            <label>
              Dataset name
              <input
                value={datasetName}
                onChange={(event) => setDatasetName(event.target.value)}
                required
                minLength={2}
                maxLength={80}
              />
            </label>
            <label className="file-input">
              <UploadSimple size={26} />
              <strong>
                {datasetFile?.name || "Choose a JSONL or JSON file"}
              </strong>
              <span>Up to 8 MB · 30–10,000 examples</span>
              <input
                type="file"
                accept=".jsonl,.json"
                onChange={(event) => setDatasetFile(event.target.files?.[0])}
                required
              />
            </label>
            <label>
              License or usage terms
              <input
                value={datasetLicense}
                onChange={(event) => setDatasetLicense(event.target.value)}
                required
                minLength={2}
              />
            </label>
            <label className="check-label">
              <input
                type="checkbox"
                checked={synthetic}
                onChange={(event) => setSynthetic(event.target.checked)}
              />
              These examples are synthetic.
            </label>
            <label className="check-label">
              <input
                type="checkbox"
                checked={permission}
                onChange={(event) => setPermission(event.target.checked)}
                required
              />
              I have permission to use these examples for training.
            </label>
            <div className="modal-footer">
              <span>Deduplicated and split locally</span>
              <button
                className="button primary"
                disabled={busy || !datasetFile || !permission}
              >
                <CheckCircle size={16} />
                Validate & import
              </button>
            </div>
          </form>
        </Modal>
      )}
    </div>
  );
}

function Summary({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <article>
      <div className="summary-label">
        {icon}
        {label}
      </div>
      <strong>{value}</strong>
      <p>{sub}</p>
    </article>
  );
}

function Status({ value }: { value: string }) {
  return (
    <span className={`status status-${value}`}>
      {value === "succeeded" ? (
        <CheckCircle size={13} weight="fill" />
      ) : value === "failed" ? (
        <WarningCircle size={13} />
      ) : value === "queued" ? (
        <Clock size={13} />
      ) : !terminal.includes(value) ? (
        <CircleNotch className="spin" size={13} />
      ) : (
        <Stop size={13} />
      )}
      {value === "succeeded"
        ? "Completed"
        : value.charAt(0).toUpperCase() + value.slice(1)}
    </span>
  );
}

function Modal({
  title,
  children,
  onClose,
  error,
}: {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
  error: string;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  return (
    <dialog
      ref={dialog}
      onCancel={onClose}
      className="modal"
      aria-label={title}
    >
      <div className="modal-heading">
        <h2>{title}</h2>
        <button
          className="icon-button"
          aria-label="Close dialog"
          onClick={onClose}
        >
          <X size={19} />
        </button>
      </div>
      {error && (
        <div className="banner error" role="alert">
          {error}
        </div>
      )}
      {children}
    </dialog>
  );
}
