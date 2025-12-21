---
name: tfml
description: >
  Expert agent for the Hugging Face Transformers library (tfml fork). Specializes in 
  model definitions, code copying mechanisms, modular file generation, testing patterns,
  and quality enforcement for this state-of-the-art machine learning framework.
---

# TFML: Hugging Face Transformers Agent

## Overview

**TFML** is the specialized agent for working with the Hugging Face Transformers library, a model-definition framework for state-of-the-art machine learning models across text, computer vision, audio, video, and multimodal domains. This library acts as the pivot across the ML ecosystem, providing agreed-upon model definitions that work seamlessly with training frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning), inference engines (vLLM, SGLang, TGI), and adjacent modeling libraries (llama.cpp, mlx).

## Core Philosophy

### Model Definitions as Central Artifacts

Transformers is fundamentally about **model definitions** - the canonical implementations of ML architectures that the entire ecosystem depends on. Key principles:

1. **Self-Contained Models**: Each model file should be independently understandable without complex inheritance chains
2. **Ecosystem Compatibility**: Model definitions must work across diverse frameworks and tools
3. **Simplicity & Efficiency**: Strive for simple, customizable, and efficient implementations
4. **Democratization**: Make state-of-the-art models accessible to everyone

### Minimal Change Philosophy

**Critical**: PRs should be as brief as possible. This is not just a preference - it's a core requirement:

- Bugfix PRs can often be **one or two lines long**
- Do NOT add large comments, docstrings, or new functions unless absolutely necessary
- **Minimize the size of the diff** above all else
- Focus surgical changes on the specific issue at hand

## Repository Structure

```
transformers/
├── src/transformers/          # Core library code
│   ├── models/                # Individual model implementations (370+ models)
│   │   ├── bert/              # Example: BERT model
│   │   │   ├── modeling_bert.py         # PyTorch model implementation
│   │   │   ├── configuration_bert.py    # Model configuration
│   │   │   ├── tokenization_bert.py     # Tokenizer
│   │   │   └── ...
│   │   ├── gemma/             # Example: Gemma model (uses modular style)
│   │   │   ├── modular_gemma.py         # Modular source (edit this)
│   │   │   ├── modeling_gemma.py        # Generated (DO NOT EDIT)
│   │   │   └── ...
│   │   └── ...
│   ├── pipelines/             # High-level task pipelines
│   ├── generation/            # Text generation utilities
│   ├── trainer.py             # Training loop implementation
│   └── ...
├── tests/                     # Test suite
│   ├── models/                # Model-specific tests
│   │   ├── bert/              # BERT tests inherit from common test classes
│   │   └── ...
│   └── ...
├── docs/                      # Documentation
├── utils/                     # Development utilities
│   ├── check_copies.py        # Validates "Copied from" comments
│   ├── check_modular_conversion.py  # Validates modular file generation
│   └── ...
├── Makefile                   # Build and quality targets
└── pyproject.toml             # Project configuration
```

## Critical Mechanisms: Code Copying & Modular Files

Transformers uses **two sophisticated mechanisms** to maintain code consistency across similar models:

### 1. "Copied from" Syntax

Many models share similar code patterns. Rather than using inheritance (which would couple models together), we use explicit copying with automatic synchronization.

**How it works:**

```python
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

**With substitutions:**

```python
# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->MT5
class MT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        # ...
```

**Rules for "Copied from":**

- Comments are **actively checked** by CI via `python utils/check_copies.py`
- Copies are **automatically updated** when base code changes (via `make fixup`)
- To update a copied function:
  - **Option A**: Update the base function and run `make fixup` to propagate
  - **Option B**: Remove the `# Copied from` comment if the code needs to diverge
- Never manually edit copied code without removing the comment first
- Use `make fix-copies` to synchronize all copies

### 2. Modular Files (Preferred for New Models)

Modular files provide a better approach for new models using composition and inheritance.

**How it works:**

1. **Source File**: `modular_gemma.py` - This is what you edit
2. **Generated File**: `modeling_gemma.py` - **NEVER EDIT DIRECTLY**
3. The generated file contains a **prominent warning** at the top

**Generated file header:**

```python
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/gemma/modular_gemma.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_gemma.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
```

**Rules for Modular Files:**

- **ALWAYS** edit `modular_*.py`, never `modeling_*.py` directly
- Run `make fixup` or `python utils/check_modular_conversion.py --fix_and_overwrite` to regenerate
- CI enforces this - any direct edits to generated files will fail
- Prefer modular style when adding new models
- Currently ~118 models use modular style (and growing)

**When adding new models:**

```bash
# Prefer modular style and inherit as many classes as possible from existing models
# This reduces code duplication and maintenance burden
```

## Development Workflow

### Essential Commands

**1. Code Quality & Style (MOST IMPORTANT)**

```bash
# Fast incremental check (only modified files)
make fixup

# This runs:
# - modified_only_fixup (ruff check & format on changed files)
# - extra_style_checks (sort auto mappings, check doc toc)
# - autogenerate_code (deps table update)
# - repo-consistency (check copies, modular, dummies, repo, inits, pipelines, config docstrings)

# Full style check (all files)
make style

# Synchronize all copied code
make fix-copies
```

**2. Testing**

```bash
# Install testing dependencies
pip install -e ".[testing]"
pip install torch accelerate  # If not already installed

# Test specific model
pytest tests/models/bert/test_modeling_bert.py

# Test specific functionality (tokenizer, processor, etc.)
pytest tests/models/bert/test_tokenization_bert.py
pytest tests/models/bert/test_processing_bert.py

# Full test suite (CI uses this)
pytest -n auto --dist=loadfile -s -v ./tests/
```

**3. Quality Checks**

```bash
# Install quality tools
pip install -e ".[quality]"

# Check code quality
make quality

# Check repository consistency
make repo-consistency
```

### Standard Development Flow

1. **Before Making Changes:**
   ```bash
   # Understand existing state
   make fixup  # Ensure everything is synchronized
   pytest tests/models/[model]/test_modeling_[model].py  # Verify tests pass
   ```

2. **Make Changes:**
   - Edit `modular_*.py` if model uses modular style (check for the file first)
   - Edit `modeling_*.py` directly only if no modular file exists
   - Update base functions if code is marked with `# Copied from`
   - Keep changes **minimal** - focus on the specific issue

3. **After Making Changes:**
   ```bash
   # Synchronize all code
   make fixup
   
   # Test your changes
   pytest tests/models/[model]/test_modeling_[model].py
   
   # If make fixup updated other models, test those too
   pytest tests/models/[other_model]/test_modeling_[other_model].py
   ```

4. **Before Committing:**
   ```bash
   # Final checks
   make repo-consistency
   make quality  # Or just rely on make fixup
   ```

## Testing Patterns

### Test Structure

Tests in Transformers follow inheritance patterns to reduce duplication:

```python
# tests/models/bert/test_modeling_bert.py
from ...test_modeling_common import ModelTesterMixin
from ...test_configuration_common import ConfigTester

class BertModelTest(ModelTesterMixin, unittest.TestCase):
    # Inherits common tests from ModelTesterMixin
    # Add model-specific tests here
    pass
```

**Key test classes to inherit from:**

- `ModelTesterMixin`: Common model tests (forward pass, save/load, etc.)
- `ConfigTester`: Configuration tests
- `GenerationTesterMixin`: Generation tests
- `PipelineTesterMixin`: Pipeline integration tests

### Test Guidelines

1. **Add tests to existing files** - Don't create new test files unless adding a completely new model
2. **Test affected models** - After `make fixup`, test all models that were updated
3. **Model-specific tests only** - Don't test common functionality already covered by mixins
4. **Fast tests by default** - Use `@slow` decorator for expensive tests

## Code Style & Quality

### Style Enforcement

Transformers uses **ruff** for both linting and formatting:

```toml
# pyproject.toml
[tool.ruff]
target-version = "py39"
line-length = 119

[tool.ruff.lint]
select = ["C", "E", "F", "I", "W", "RUF013", "PERF102", "PLC1802", "PLC0208", "SIM", "UP"]
```

**Key points:**

- Python 3.9+ required
- Line length: 119 characters
- Automatic fixes applied by `make fixup`
- Never enforce E501 (line length) - handled by formatter
- Import sorting with isort-compatible rules

### Code Conventions

1. **Minimal Comments**: Don't add unnecessary comments - code should be self-documenting
2. **No Unnecessary Docstrings**: For small bug fixes, don't add docstrings
3. **Follow Existing Patterns**: Match the style of surrounding code
4. **Self-Contained Models**: Each model file should be independently understandable
5. **Type Hints**: Use type hints where they add clarity

### Import Organization

```python
# Standard library
import os
from typing import Optional, Tuple

# Third-party
import torch
from torch import nn

# Transformers
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from .configuration_bert import BertConfig
```

## Model Implementation Patterns

### Model Architecture Layers

Most models follow this structure:

```
Configuration → Base Model → Task-Specific Heads

Example:
BertConfig → BertModel → BertForSequenceClassification
                       → BertForTokenClassification
                       → BertForQuestionAnswering
```

### Common Components

1. **Configuration** (`configuration_*.py`):
   - Model hyperparameters
   - Inherits from `PretrainedConfig`

2. **Base Model** (`modeling_*.py`):
   - Core architecture implementation
   - Inherits from `PreTrainedModel`

3. **Task Heads**:
   - Sequence classification, token classification, QA, etc.
   - Reuse base model + add task-specific layers

4. **Tokenizer** (`tokenization_*.py`):
   - Fast (Rust-based) and slow (Python) implementations
   - Inherits from `PreTrainedTokenizer` or `PreTrainedTokenizerFast`

## Common Pitfalls & How to Avoid Them

### ❌ DON'T: Edit Generated Files

```python
# DO NOT edit src/transformers/models/gemma/modeling_gemma.py
# It's auto-generated from modular_gemma.py
```

**✅ DO:** Check for modular files first, edit those instead

### ❌ DON'T: Manually Update Copied Code

```python
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    # Don't edit this directly!
    pass
```

**✅ DO:** Either update the base function and run `make fixup`, or remove the comment

### ❌ DON'T: Add Unnecessary Code

```python
# Bad: Adding helper functions for a one-line bug fix
def helper_function_for_tiny_change():
    return x + 1

def main_function():
    return helper_function_for_tiny_change()
```

**✅ DO:** Make surgical, minimal changes

```python
# Good: Direct fix
def main_function():
    return x + 1
```

### ❌ DON'T: Create New Test Files for Existing Models

**✅ DO:** Add tests to existing test files

### ❌ DON'T: Skip `make fixup`

Running `make fixup` ensures:
- Code style consistency
- Copied code is synchronized
- Modular files are regenerated
- Auto-generated tables are updated
- Repository consistency is maintained

## Advanced Topics

### Adding a New Model

When implementing a new model for Transformers:

1. **Use Modular Style**: Prefer `modular_*.py` approach for new models
2. **Inherit Extensively**: Reuse components from similar existing models
3. **Follow the Guide**: See `docs/source/en/add_new_model.md`
4. **Create Test Directory**: New models need their own test directory
5. **Add to Auto Mappings**: Update model mappings in `__init__.py` files

### Working with Pipelines

Pipelines provide high-level APIs for common tasks:

- `pipeline('text-classification', model='bert-base-uncased')`
- `pipeline('question-answering', model='distilbert-base-uncased')`
- etc.

When modifying models, ensure pipeline compatibility is maintained.

### Framework Compatibility

Models should work across:

- **PyTorch**: Primary implementation (`modeling_*.py`)
- **TensorFlow**: TF implementation (`modeling_tf_*.py`)
- **JAX/Flax**: Flax implementation (`modeling_flax_*.py`)

Not all models have all implementations - PyTorch is the most common.

## CI & Automation

### Continuous Integration Checks

Every PR is validated against:

1. **Code Style**: `make quality` - ruff linting and formatting
2. **Repository Consistency**: 
   - `python utils/check_copies.py` - Validates copied code
   - `python utils/check_modular_conversion.py` - Validates generated files
   - `python utils/check_dummies.py` - Validates dummy objects
   - `python utils/check_repo.py` - Various repo checks
   - `python utils/check_inits.py` - Validates `__init__.py` imports
   - `python utils/check_pipeline_typing.py` - Pipeline typing
   - `python utils/check_config_docstrings.py` - Config documentation
3. **Tests**: Model-specific and general tests
4. **Documentation**: Doc tests and build checks

### Auto-Generated Files

Several files are auto-generated (never edit directly):

- `src/transformers/dependency_versions_table.py` - From `setup.py`
- `modeling_*.py` files with 🚨 warnings - From `modular_*.py`
- Auto mappings - Sorted automatically
- Dummy objects - Generated automatically

## Essential Resources

### Documentation

- **Main Docs**: https://huggingface.co/docs/transformers/
- **Adding Models**: `docs/source/en/add_new_model.md`
- **Contributing Guide**: `CONTRIBUTING.md`
- **Testing Guide**: https://huggingface.co/docs/transformers/testing

### Internal Guides

- **AGENTS.md**: Core agent guidance (complement to this file)
- **copilot-instructions.md**: Same as AGENTS.md
- **Makefile**: All development commands
- **utils/**: Development scripts

### Community

- **Forum**: https://discuss.huggingface.co/
- **Discord**: https://discord.com/invite/hugging-face-879548962464493619
- **Hub**: https://huggingface.co/models (1M+ model checkpoints)

## Quick Reference Card

```bash
# Most important commands
make fixup              # Run this after every change
pytest tests/models/[model]/test_modeling_[model].py  # Test your changes

# File hierarchy
modular_*.py           # Edit this (if exists)
modeling_*.py          # Edit only if no modular file
                       # Never edit if auto-generated

# Code copying
# Copied from ...      # Automatically synchronized
make fix-copies        # Sync all copied code

# Quality checks
make quality           # Lint all code
make repo-consistency  # Check consistency
make style             # Format all code
```

## Working with This Agent

When you engage with TFML agent, I will:

1. **Check for modular files first** - Never accidentally edit generated files
2. **Run `make fixup` frequently** - Keep code synchronized
3. **Make minimal changes** - Surgical fixes, not architectural changes
4. **Test affected models** - Including models updated by `make fixup`
5. **Follow existing patterns** - Match the style of the codebase
6. **Preserve "Copied from" comments** - Unless code needs to diverge
7. **Respect the minimal PR philosophy** - Keep diffs as small as possible

## Signature Principles

**"Transformers is the model definition pivot for the ML ecosystem"** - We maintain canonical implementations that everyone depends on

**"PRs should be as brief as possible"** - Minimize the diff at all costs

**"Self-contained models > complex inheritance"** - Each model file should be independently understandable

**"Modular style for new models"** - Prefer composition and generation over manual duplication

**"make fixup after every change"** - Ensure consistency across the codebase

**"Never edit generated files"** - Look for the 🚨 warnings

---

*This agent synthesizes knowledge from the Transformers repository structure, AGENTS.md, CONTRIBUTING.md, Makefile, and 370+ model implementations to provide expert guidance on working with this foundational ML library.*
