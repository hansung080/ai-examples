#!/usr/bin/env bash

S_RESET=$'\033[0m'
S_BOLD=$'\033[1m'

PROJECT_ROOT="$(dirname "$(readlink -f "$0")")"
MYPY=("$PROJECT_ROOT/.venv/bin/mypy" --strict)

if (($# > 0)); then
  "${MYPY[@]}" "$@"
else
  shopt -u dotglob

  # A while loop is used instead of find -exec to print the success message only once at the end.
  # find "$PROJECT_ROOT/"* -type f -name '*.py' -exec "${MYPY[@]}" {} \;

  COUNT=0
  ERRORED=false
  while IFS= read -rd '' FILE; do
    if ! "${MYPY[@]}" "$FILE" &> /dev/null; then
      ERRORED=true
      "${MYPY[@]}" "$FILE" >&2
    fi
    ((++COUNT))
  done < <(find "$PROJECT_ROOT/"* -type f -name '*.py' -print0)

  "$ERRORED" && exit 1

  if ((COUNT == 0)); then
    echo "There are no .py[i] files in directory '$PROJECT_ROOT'"
  elif ((COUNT == 1)); then
    echo "${S_BOLD}Success: no issues found in $COUNT source file${S_RESET}"
  else
    echo "${S_BOLD}Success: no issues found in $COUNT source files${S_RESET}"
  fi
fi
