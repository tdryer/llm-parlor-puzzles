ruff := "uv run ruff"

default:
  just --list --unsorted

format:
    {{ruff}} format
    {{ruff}} check --select I --fix

format-check:
    {{ruff}} format --check
    {{ruff}} check --extend-select I

lint:
    {{ruff}} check

pre-commit: format-check lint

install-pre-commit:
    #!/bin/bash
    HOOK_PATH=.git/hooks/pre-commit
    cat > "$HOOK_PATH" << 'EOF'
    #!/bin/bash
    echo "Running 'just pre-commit'..."
    if ! just pre-commit; then
      echo "Commit aborted: 'just pre-commit' failed."
      exit 1
    fi
    EOF
    chmod +x "$HOOK_PATH"
    echo "Pre-commit hook created successfully."
