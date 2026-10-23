$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
# The credential helper must be on PATH as well as docker.exe.
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $dockerDirectories = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin",
        "$env:ProgramFiles\Docker\Docker\resources\bin"
    )
    foreach ($directory in $dockerDirectories) {
        if (Test-Path (Join-Path $directory 'docker.exe')) {
            $env:PATH = "$directory;$env:PATH"
            break
        }
    }
}
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker CLI was not found. Install/start Docker Desktop and reopen PowerShell.'
}
$dockerDirectory = Split-Path (Get-Command docker).Source -Parent
$env:PATH = "$dockerDirectory;$env:PATH"
docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable. Start Docker Desktop.' }
docker compose config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Compose configuration validation failed.' }
foreach ($service in @('backend', 'frontend')) {
    docker compose build $service
    if ($LASTEXITCODE -ne 0) { throw "Image build failed: $service" }
}
docker image ls --filter 'reference=modular-rag-*:local'
