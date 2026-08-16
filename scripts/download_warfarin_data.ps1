$ErrorActionPreference = "Stop"
$destination = Join-Path $PSScriptRoot "..\data\warfarin_dat.csv"
$url = "https://holford.fmhs.auckland.ac.nz/research/nlmixr/warfarin/warfarin_dat.csv"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
Invoke-WebRequest -Uri $url -OutFile $destination
Write-Output "Downloaded public warfarin dataset to $destination"