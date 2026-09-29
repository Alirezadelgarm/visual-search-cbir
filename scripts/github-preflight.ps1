$ErrorActionPreference = "Stop"

$forbiddenPatterns = @(
    '^data/',
    '^\.venv/',
    '\.joblib$',
    '\.pkl$',
    '__pycache__',
    '\.pyc$'
)

$staged = git diff --cached --name-only
foreach ($file in $staged) {
    foreach ($pattern in $forbiddenPatterns) {
        if ($file -match $pattern) {
            throw "Forbidden generated/local file is staged: $file"
        }
    }
}

$largeFiles = git ls-files | ForEach-Object {
    if (Test-Path $_) {
        $item = Get-Item $_
        if ($item.Length -gt 25MB) { $_ }
    }
}
if ($largeFiles) {
    throw "Large tracked files detected:`n$($largeFiles -join "`n")"
}

Write-Host "GitHub preflight passed." -ForegroundColor Green
