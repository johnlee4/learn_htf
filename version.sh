#!/bin/bash

# Get the latest tag
latest_tag=$(git describe --tags --abbrev=0)
IFS='.' read -r major main develop <<< "$latest_tag"

# Check which part to increment based on branch name
branch=$(git rev-parse --abbrev-ref HEAD)

if [[ "$branch" == "main" ]]; then
  # Increment main version, reset develop
  main=$((main + 1))
  develop=0
elif [[ "$branch" == "develop" ]]; then
  # Increment develop version
  develop=$((develop + 1))
else
  echo "Not on main or develop branch, version not updated."
  exit 0
fi

# Create the new version tag
new_tag="${major}.${main}.${develop}"
git tag "$new_tag"
git push origin "$new_tag"

echo "New tag created: $new_tag"
