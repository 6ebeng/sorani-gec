# Publish docs/wiki/*.md to the GitHub wiki (6ebeng/sorani-gec.wiki.git).
#
# One-time prerequisite: GitHub only creates the wiki git repo after the first
# page exists. Go to https://github.com/6ebeng/sorani-gec/wiki , click
# "Create the first page", save the default page once, then run this script.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File docs/wiki/publish-wiki.ps1

$ErrorActionPreference = "Stop"
$wikiSrc  = $PSScriptRoot
$wikiRepo = "https://github.com/6ebeng/sorani-gec.wiki.git"
$tmp      = Join-Path ([System.IO.Path]::GetTempPath()) "sorani-gec-wiki"

if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }

git clone $wikiRepo $tmp
if ($LASTEXITCODE -ne 0) {
    Write-Error "Clone failed. Create the first wiki page on GitHub, then re-run."
}

# Sync: copy all .md pages (script itself excluded), remove stale pages
Get-ChildItem $tmp -Filter *.md | Remove-Item -Force
Copy-Item (Join-Path $wikiSrc "*.md") $tmp -Force

Push-Location $tmp
git add -A
git -c core.safecrlf=false commit -m "docs: sync wiki from docs/wiki ($(Get-Date -Format yyyy-MM-dd))"
git push origin master 2>$null; if ($LASTEXITCODE -ne 0) { git push origin main }
Pop-Location

Remove-Item $tmp -Recurse -Force
Write-Host "Wiki published: https://github.com/6ebeng/sorani-gec/wiki"
