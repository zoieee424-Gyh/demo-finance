$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

& "D:\AI\soft\conda\envs\python3.11\python.exe" backend\app\main.py
