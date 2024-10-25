#!/bin/bash

# Get the latest tag
latest_tag=$(git tag --sort=-v:refname | head -n 1)
IFS='.' read -r major master develop <<< "$latest_tag"

# Check which part to increment based on branch name
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" == "master" ]]; then
  # Increment master version, reset develop
  master=$((master + 1))
  develop=0
elif [[ "$branch" == "develop" ]]; then
  # Increment develop version
  develop=$((develop + 1))
else
  echo "Not on master or develop branch, version not updated."
  exit 0
fi
# Create the new version tag
new_tag="${major}.${master}.${develop}"
git tag "$new_tag"
git push origin "$new_tag"

echo "New tag created: $new_tag"
