---
name: hugging-face-evaluation-manager
description: Add and manage evaluation results in Hugging Face model cards. Supports extracting eval tables from README content and importing scores from Artificial Analysis API. Works with the model-index metadata format.
---

# Overview
This skill provides tools to add structured evaluation results to Hugging Face model cards. It supports two primary methods for adding evaluation data: extracting existing evaluation tables from README content and importing benchmark scores from Artificial Analysis.

## Integration with HF Ecosystem
- **Model Cards**: Updates model-index metadata for leaderboard integration
- **Artificial Analysis**: Direct API integration for benchmark imports
- **Papers with Code**: Compatible with their model-index specification
- **Jobs**: Run evaluations directly on Hugging Face Jobs with `uv` integration

# Version
1.2.0

# Dependencies
- huggingface_hub>=0.26.0
- markdown-it-py>=3.0.0
- python-dotenv>=1.2.1
- pyyaml>=6.0.3
- requests>=2.32.5
- inspect-ai>=0.3.0
- re (built-in)

# IMPORTANT: Using This Skill

**Always run `--help` to get guidance on table extraction and YAML generation:**
```bash
python scripts/evaluation_manager.py --help
python scripts/evaluation_manager.py inspect-tables --help
python scripts/evaluation_manager.py extract-readme --help
```

The `--help` output includes workflow guidance for converting tables to YAML.

# Core Capabilities

## 1. Inspect and Extract Evaluation Tables from README
- **Inspect Tables**: Use `inspect-tables` to see all tables in a README with their structure, columns, and suggested extraction commands
- **Parse Markdown Tables**: Accurate parsing using markdown-it-py (ignores code blocks and examples)
- **Table Selection**: Use `--table N` to extract from a specific table (required when multiple tables exist)
- **Format Detection**: Recognize common formats (benchmarks as rows, columns, or comparison tables with multiple models)
- **Column Matching**: Automatically identify model columns, with `--model-name-override` for comparison tables
- **YAML Generation**: Convert selected table to model-index YAML format

## 2. Import from Artificial Analysis
- **API Integration**: Fetch benchmark scores directly from Artificial Analysis
- **Automatic Formatting**: Convert API responses to model-index format
- **Metadata Preservation**: Maintain source attribution and URLs
- **PR Creation**: Automatically create pull requests with evaluation updates

## 3. Model-Index Management
- **YAML Generation**: Create properly formatted model-index entries
- **Merge Support**: Add evaluations to existing model cards without overwriting
- **Validation**: Ensure compliance with Papers with Code specification
- **Batch Operations**: Process multiple models efficiently

## 4. Run Evaluations on HF Jobs
- **Inspect-AI Integration**: Run standard evaluations using the `inspect-ai` library
- **UV Integration**: Seamlessly run Python scripts with ephemeral dependencies on HF infrastructure
- **Zero-Config**: No Dockerfiles or Space management required
- **Hardware Selection**: Configure CPU or GPU hardware for the evaluation job
- **Secure Execution**: Handles API tokens safely via secrets passed through the CLI

# Usage Instructions

The skill includes Python scripts in `scripts/` to perform operations.

### Prerequisites
- Install dependencies: `uv add huggingface_hub python-dotenv pyyaml inspect-ai`
- Set `HF_TOKEN` environment variable with Write-access token
- For Artificial Analysis: Set `AA_API_KEY` environment variable
- Activate virtual environment: `source .venv/bin/activate`

### Method 1: Extract from README

Extract evaluation tables from a model's existing README and convert to model-index YAML.

#### Recommended Workflow: Inspect Tables First

**Step 1: Inspect the tables** to see structure and get the extraction command:
```bash
python scripts/evaluation_manager.py inspect-tables --repo-id "allenai/OLMo-7B"
```

This outputs:
```
======================================================================
Tables found in README for: allenai/OLMo-7B
======================================================================

## Table 3
   Format: comparison
   Rows: 14

   Columns (6):
      [1] [Llama 7B](...)
      [2] [Llama 2 7B](...)
      [5] **OLMo 7B** (ours)  ~ partial match

   Sample rows (first column):
      - arc_challenge
      - arc_easy
      - boolq

   ⚠ No exact match. Best candidate: **OLMo 7B** (ours)

   Suggested command:
      python scripts/evaluation_manager.py extract-readme \
        --repo-id "allenai/OLMo-7B" \
        --table 3 \
        --model-name-override "**OLMo 7B** (ours)" \
        --dry-run
```

**Step 2: Copy and run the suggested command** (with `--dry-run` to preview YAML):
```bash
python scripts/evaluation_manager.py extract-readme \
  --repo-id "allenai/OLMo-7B" \
  --table 3 \
  --model-name-override "**OLMo 7B** (ours)" \
  --dry-run
```

**Step 3: Verify the YAML output** - check benchmark names and values match the README

**Step 4: Apply changes** - remove `--dry-run` and optionally add `--create-pr`:
```bash
python scripts/evaluation_manager.py extract-readme \
  --repo-id "allenai/OLMo-7B" \
  --table 3 \
  --model-name-override "**OLMo 7B** (ours)" \
  --create-pr
```

#### Key Flags

- `--table N`: **Required when multiple tables exist.** Specifies which table to extract (1-indexed, matches `inspect-tables` output)
- `--model-name-override`: Column header text for comparison tables (e.g., `"**OLMo 7B** (ours)"`)
- `--dry-run`: Preview YAML without making changes
- `--create-pr`: Create a pull request instead of direct push

#### Supported Table Formats

**Format 1: Benchmarks as Rows**
```markdown
| Benchmark | Score |
|-----------|-------|
| MMLU      | 85.2  |
| HumanEval | 72.5  |
```

**Format 2: Benchmarks as Columns**
```markdown
| MMLU | HumanEval | GSM8K |
|------|-----------|-------|
| 85.2 | 72.5      | 91.3  |
```

**Format 3: Multiple Metrics**
```markdown
| Benchmark | Accuracy | F1 Score |
|-----------|----------|----------|
| MMLU      | 85.2     | 0.84     |
```

**Format 4: Transposed Tables (Models as Rows)**
```markdown
| Model          | MMLU | HumanEval | GSM8K | ARC  |
|----------------|------|-----------|-------|------|
| GPT-4          | 86.4 | 67.0      | 92.0  | 96.3 |
| Claude-3       | 86.8 | 84.9      | 95.0  | 96.4 |
| **Your-Model** | 85.2 | 72.5      | 91.3  | 95.8 |
```

In this format, the script will:
- Detect that models are in rows (first column) and benchmarks in columns (header)
- Find the row matching your model name (handles bold/markdown formatting)
- Extract all benchmark scores from that specific row only

#### Validating Extraction Results

**CRITICAL**: Always validate extracted results before creating a PR or pushing changes.

After running `extract-readme`, you MUST:

1. **Use `--dry-run` first** to preview the extraction:
```bash
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  --dry-run
```

2. **Manually verify the output**:
   - Check that the correct model's scores were extracted (not other models)
   - Verify benchmark names are correct
   - Confirm all expected benchmarks are present
   - Ensure numeric values match the README exactly

3. **For transposed tables** (models as rows):
   - Verify only ONE model's row was extracted
   - Check that it matched the correct model name
   - Look for warnings like "Could not find model 'X' in transposed table"
   - If scores from multiple models appear, the table format was misdetected

4. **Compare against the source**:
   - Open the model README in a browser
   - Cross-reference each extracted score with the table
   - Verify no scores are mixed from different rows/columns

5. **Common validation failures**:
   - **Multiple models extracted**: Wrong table format detected
   - **Missing benchmarks**: Column headers not recognized
   - **Wrong scores**: Matched wrong model row or column
   - **Empty metrics list**: Table not detected or parsing failed

**Example validation workflow**:
```bash
# Step 1: Dry run to preview
python scripts/evaluation_manager.py extract-readme \
  --repo-id "allenai/Olmo-3-1125-32B" \
  --dry-run

# Step 2: If model name not found in table, script shows available models
# ⚠ Could not find model 'Olmo-3-1125-32B' in transposed table
#
# Available models in table:
#   1. **Open-weight Models**
#   2. Qwen-2.5-32B
#   ...
#   12. **Olmo 3-32B**
#
# Please select the correct model name from the list above.

# Step 3: Re-run with the correct model name
python scripts/evaluation_manager.py extract-readme \
  --repo-id "allenai/Olmo-3-1125-32B" \
  --model-name-override "**Olmo 3-32B**" \
  --dry-run

# Step 4: Review the YAML output carefully
# Verify: Are these all benchmarks for Olmo-3-32B ONLY?
# Verify: Do the scores match the README table?

# Step 5: If validation passes, create PR
python scripts/evaluation_manager.py extract-readme \
  --repo-id "allenai/Olmo-3-1125-32B" \
  --model-name-override "**Olmo 3-32B**" \
  --create-pr

# Step 6: Validate the model card after update
python scripts/evaluation_manager.py show \
  --repo-id "allenai/Olmo-3-1125-32B"
```

### Method 2: Import from Artificial Analysis

Fetch benchmark scores from Artificial Analysis API and add them to a model card.

**Basic Usage:**
```bash
AA_API_KEY="your-api-key" python scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "username/model-name"
```

**With Environment File:**
```bash
# Create .env file
echo "AA_API_KEY=your-api-key" >> .env
echo "HF_TOKEN=your-hf-token" >> .env

# Run import
python scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "username/model-name"
```

**Create Pull Request:**
```bash
python scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "username/model-name" \
  --create-pr
```

### Method 3: Run Evaluation Job

Submit an evaluation job on Hugging Face infrastructure using the `hf jobs uv run` CLI.

**Direct CLI Usage:**
```bash
HF_TOKEN=$HF_TOKEN \
hf jobs uv run hf_model_evaluation/scripts/inspect_eval_uv.py \
  --flavor cpu-basic \
  --secret HF_TOKEN=$HF_TOKEN \
  -- --model "meta-llama/Llama-2-7b-hf" \
     --task "mmlu"
```

**GPU Example (A10G):**
```bash
HF_TOKEN=$HF_TOKEN \
hf jobs uv run hf_model_evaluation/scripts/inspect_eval_uv.py \
  --flavor a10g-small \
  --secret HF_TOKEN=$HF_TOKEN \
  -- --model "meta-llama/Llama-2-7b-hf" \
     --task "gsm8k"
```

**Python Helper (optional):**
```bash
python scripts/run_eval_job.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --task "mmlu" \
  --hardware "t4-small"
```

### Commands Reference

**List Available Commands:**
```bash
python scripts/evaluation_manager.py --help
```

**Inspect Tables (start here):**
```bash
python scripts/evaluation_manager.py inspect-tables \
  --repo-id "username/model-name"
```
Shows all tables in the README with:
- Table format (simple, comparison, transposed)
- Column headers with model match indicators
- Sample rows from first column
- **Ready-to-use `extract-readme` command** with correct `--table` and `--model-name-override`

Run `inspect-tables --help` to see the full workflow.

**Extract from README:**
```bash
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  [--table N] \
  [--model-name-override "Column Header"] \
  [--task-type "text-generation"] \
  [--dataset-name "Custom Benchmarks"] \
  [--dry-run] \
  [--create-pr]
```

Key flags:
- `--table N`: Table number from `inspect-tables` output (required if multiple tables)
- `--model-name-override`: Exact column header for comparison tables
- `--dry-run`: Preview YAML output without applying

**Import from Artificial Analysis:**
```bash
python scripts/evaluation_manager.py import-aa \
  --creator-slug "creator-name" \
  --model-name "model-slug" \
  --repo-id "username/model-name" \
  [--create-pr]
```

**View Current Evaluations:**
```bash
python scripts/evaluation_manager.py show \
  --repo-id "username/model-name"
```

**Validate Model-Index:**
```bash
python scripts/evaluation_manager.py validate \
  --repo-id "username/model-name"
```

**Run Evaluation Job:**
```bash
hf jobs uv run hf_model_evaluation/scripts/inspect_eval_uv.py \
  --flavor "cpu-basic|t4-small|..." \
  --secret HF_TOKEN=$HF_TOKEN \
  -- --model "model-id" \
     --task "task-name"
```

or use the Python helper:

```bash
python scripts/run_eval_job.py \
  --model "model-id" \
  --task "task-name" \
  --hardware "cpu-basic|t4-small|..."
```

### Model-Index Format

The generated model-index follows this structure:

```yaml
model-index:
  - name: Model Name
    results:
      - task:
          type: text-generation
        dataset:
          name: Benchmark Dataset
          type: benchmark_type
        metrics:
          - name: MMLU
            type: mmlu
            value: 85.2
          - name: HumanEval
            type: humaneval
            value: 72.5
        source:
          name: Source Name
          url: https://source-url.com
```

WARNING: Do not use markdown formatting in the model name. Use the exact name from the table. Only use urls in the source.url field.

### Advanced Usage

**Extract Multiple Tables:**
```bash
# The script automatically detects and processes all evaluation tables
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  --merge-tables
```

**Custom Metric Mapping:**
```bash
# Use a JSON file to map column names to metric types
python scripts/evaluation_manager.py extract-readme \
  --repo-id "username/model-name" \
  --metric-mapping "$(cat metric_mapping.json)"
```

Example `metric_mapping.json`:
```json
{
  "MMLU": {"type": "mmlu", "name": "Massive Multitask Language Understanding"},
  "HumanEval": {"type": "humaneval", "name": "Code Generation (HumanEval)"},
  "GSM8K": {"type": "gsm8k", "name": "Grade School Math"}
}
```

**Batch Processing:**
```bash
# Process multiple models from a list
while read repo_id; do
  python scripts/evaluation_manager.py extract-readme --repo-id "$repo_id"
done < models.txt
```

### Error Handling
- **Table Not Found**: Script will report if no evaluation tables are detected
- **Invalid Format**: Clear error messages for malformed tables
- **API Errors**: Retry logic for transient Artificial Analysis API failures
- **Token Issues**: Validation before attempting updates
- **Merge Conflicts**: Preserves existing model-index entries when adding new ones
- **Space Creation**: Handles naming conflicts and hardware request failures gracefully

### Best Practices

1. **Always start with `inspect-tables`**: See table structure and get the correct extraction command
2. **Use `--help` for guidance**: Run `inspect-tables --help` to see the complete workflow
3. **Use `--dry-run` first**: Preview YAML output before applying changes
4. **Verify extracted values**: Compare YAML output against the README table manually
5. **Use `--table N` for multi-table READMEs**: Required when multiple evaluation tables exist
6. **Use `--model-name-override` for comparison tables**: Copy the exact column header from `inspect-tables` output
7. **Create PRs for Others**: Use `--create-pr` when updating models you don't own
8. **One model per repo**: Only add the main model's results to model-index
9. **No markdown in YAML names**: The model name field in YAML should be plain text

### Model Name Matching

When extracting evaluation tables with multiple models (either as columns or rows), the script uses **exact normalized token matching**:

- Removes markdown formatting (bold `**`, links `[]()`  )
- Normalizes names (lowercase, replace `-` and `_` with spaces)
- Compares token sets: `"OLMo-3-32B"` → `{"olmo", "3", "32b"}` matches `"**Olmo 3 32B**"` or `"[Olmo-3-32B](...)`
- Only extracts if tokens match exactly (handles different word orders and separators)
- Fails if no exact match found (rather than guessing from similar names)

**For column-based tables** (benchmarks as rows, models as columns):
- Finds the column header matching the model name
- Extracts scores from that column only

**For transposed tables** (models as rows, benchmarks as columns):
- Finds the row in the first column matching the model name
- Extracts all benchmark scores from that row only

This ensures only the correct model's scores are extracted, never unrelated models or training checkpoints. 

### Common Patterns

**Update Your Own Model:**
```bash
# Extract from README and push directly
python scripts/evaluation_manager.py extract-readme \
  --repo-id "your-username/your-model" \
  --task-type "text-generation"
```

**Update Someone Else's Model:**
```bash
# Create a PR instead of direct push
python scripts/evaluation_manager.py extract-readme \
  --repo-id "other-username/their-model" \
  --create-pr
```

**Import Fresh Benchmarks:**
```bash
# Get latest scores from Artificial Analysis
python scripts/evaluation_manager.py import-aa \
  --creator-slug "anthropic" \
  --model-name "claude-sonnet-4" \
  --repo-id "anthropic/claude-sonnet-4" \
  --create-pr
```

### Troubleshooting

**Issue**: "No evaluation tables found in README"
- **Solution**: Check if README contains markdown tables with numeric scores

**Issue**: "Could not find model 'X' in transposed table"
- **Solution**: The script will display available models. Use `--model-name-override` with the exact name from the list
- **Example**: `--model-name-override "**Olmo 3-32B**"`

**Issue**: "AA_API_KEY not set"
- **Solution**: Set environment variable or add to .env file

**Issue**: "Token does not have write access"
- **Solution**: Ensure HF_TOKEN has write permissions for the repository

**Issue**: "Model not found in Artificial Analysis"
- **Solution**: Verify creator-slug and model-name match API values

**Issue**: "Payment required for hardware"
- **Solution**: Add a payment method to your Hugging Face account to use non-CPU hardware

### Integration Examples

**Python Script Integration:**
```python
import subprocess
import os

def update_model_evaluations(repo_id, readme_content):
    """Update model card with evaluations from README."""
    result = subprocess.run([
        "python", "scripts/evaluation_manager.py",
        "extract-readme",
        "--repo-id", repo_id,
        "--create-pr"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Successfully updated {repo_id}")
    else:
        print(f"Error: {result.stderr}")
```
