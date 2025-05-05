import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("queue a real training job, export it, and run inference", async ({
  page,
  request,
}, info) => {
  test.skip(
    info.project.name === "mobile",
    "Training path is exercised once; mobile covers navigation and layout.",
  );
  await page.goto("/");
  await page
    .getByRole("button", { name: "New experiment", exact: true })
    .click();
  const dialog = page.getByRole("dialog", { name: "New experiment" });
  await dialog
    .getByLabel("Experiment name", { exact: true })
    .fill("browser-training-smoke");
  await dialog.getByLabel("Training recipe").selectOption("head-only");
  await dialog.getByLabel("Epochs", { exact: true }).fill("1");
  await dialog.getByText("Batch size, sequence length, and seed").click();
  await dialog
    .getByRole("combobox", { name: "Batch size", exact: true })
    .selectOption("32");
  await dialog.getByLabel("Sequence length").fill("32");
  await dialog.getByRole("button", { name: "Queue experiment" }).click();
  await expect(dialog).not.toBeVisible();
  let runId = "";
  await expect
    .poll(
      async () => {
        const runs = await (await request.get("/api/runs")).json();
        const run = runs.find(
          (row: { name: string }) => row.name === "browser-training-smoke",
        );
        runId = run?.id || "";
        return run?.status;
      },
      { timeout: 150_000, intervals: [1000, 2000] },
    )
    .toBe("succeeded");
  await expect(
    page.getByRole("heading", { name: "Measured after checkpoint selection" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Inference", exact: true }).click();
  await page
    .getByLabel("Support request")
    .fill("I cannot sign in to my account.");
  await page.getByRole("button", { name: "Run inference" }).click();
  await expect(page.locator(".prediction-title strong")).toBeVisible({
    timeout: 45_000,
  });
  await page.getByRole("button", { name: "Artifacts", exact: true }).click();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download model bundle" }).click();
  const file = await download;
  expect(file.suggestedFilename()).toBe(`adapterlab-${runId.slice(0, 8)}.zip`);
  expect((await readFile((await file.path())!)).subarray(0, 2).toString()).toBe(
    "PK",
  );
});

test("dataset validation errors remain visible inside the dialog", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Datasets", exact: true }).click();
  await expect(page.getByText("Groups separated across splits")).toBeVisible();
  await page
    .getByRole("button", { name: "Import dataset", exact: true })
    .click();
  const dialog = page.getByRole("dialog", { name: "Import a dataset" });
  await dialog
    .getByLabel("Dataset name", { exact: true })
    .fill("invalid-dataset");
  await dialog.locator("input[type=file]").setInputFiles({
    name: "invalid.jsonl",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify({ text: "only one example", label: "support" }),
    ),
  });
  await dialog
    .getByText("I have permission to use these examples for training.")
    .click();
  await dialog.getByRole("button", { name: "Validate & import" }).click();
  await expect(dialog.getByRole("alert")).toBeVisible();
  await page.waitForTimeout(3000);
  await expect(dialog.getByRole("alert")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
});

test("workspace navigation, registry, and responsive layout", async ({
  page,
  request,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  const rows = await (await request.get("/api/runs")).json();
  const trained =
    rows.find(
      (row: { status: string; config: { strategy: string } }) =>
        row.status === "succeeded" && row.config.strategy === "lora",
    ) || rows.find((row: { status: string }) => row.status === "succeeded");
  if (trained) {
    await page
      .getByRole("button", { name: new RegExp(`^${trained.name} `) })
      .click();
    await expect(
      page.getByRole("heading", {
        name: "Measured after checkpoint selection",
      }),
    ).toBeVisible();
  }
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  const file = `${info.project.name === "desktop" ? "dashboard" : "mobile"}.png`;
  await page.screenshot({
    path: process.env.UPDATE_DOC_SCREENSHOTS
      ? `../docs/${file}`
      : info.outputPath(file),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Datasets", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Dataset versions" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Model registry", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Completed checkpoints" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});
