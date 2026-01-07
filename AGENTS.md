<skills>

You have additional SKILLs documented in directories containing a "SKILL.md" file.

These skills are:
 - hugging-face-cli -> "skills/hugging-face-cli/SKILL.md"
 - hugging-face-datasets -> "skills/hugging-face-datasets/SKILL.md"
 - hugging-face-evaluation -> "skills/hugging-face-evaluation/SKILL.md"
 - hugging-face-jobs -> "skills/hugging-face-jobs/SKILL.md"
 - hugging-face-model-trainer -> "skills/hugging-face-model-trainer/SKILL.md"
 - hugging-face-paper-publisher -> "skills/hugging-face-paper-publisher/SKILL.md"
 - hugging-face-tool-builder -> "skills/hugging-face-tool-builder/SKILL.md"

IMPORTANT: You MUST read the SKILL.md file whenever the description of the skills matches the user intent, or may help accomplish their task. 

<available_skills>

hugging-face-cli: `Execute Hugging Face Hub operations using the hf CLI. Download models/datasets, upload files, manage repos, and run cloud compute jobs.`
hugging-face-datasets: `Create and manage datasets on Hugging Face Hub. Supports initializing repos, defining configs/system prompts, streaming row updates, and SQL-based dataset querying/transformation. Designed to work alongside HF MCP server for comprehensive dataset workflows.`
hugging-face-evaluation: `Add and manage evaluation results in Hugging Face model cards. Supports extracting eval tables from README content, importing scores from Artificial Analysis API, and running custom model evaluations with vLLM/lighteval. Works with the model-index metadata format.`
hugging-face-jobs: `Run compute jobs on Hugging Face infrastructure. Execute Python scripts, manage scheduled jobs, and monitor job status. Covers UV scripts, Docker-based jobs, hardware selection, cost estimation, and result persistence.`
hugging-face-model-trainer: `Train or fine-tune language models using TRL on Hugging Face Jobs infrastructure. Covers SFT, DPO, GRPO and reward modeling training methods, plus GGUF conversion for local deployment. Includes hardware selection, cost estimation, Trackio monitoring, and Hub persistence.`
hugging-face-paper-publisher: `Publish and manage research papers on Hugging Face Hub. Supports creating paper pages, linking papers to models/datasets, claiming authorship, and generating professional markdown-based research articles.`
hugging-face-tool-builder: `Build reusable scripts for Hugging Face API operations. Useful for chaining API calls or automating repeated tasks. Creates command line tools that can be piped and composed.`
</available_skills>

Paths referenced within SKILL folders are relative to that SKILL. For example the hugging-face-datasets `scripts/dataset_manager.py` would be referenced as `skills/hugging-face-datasets/scripts/dataset_manager.py`. 

</skills>
