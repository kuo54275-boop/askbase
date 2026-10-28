# AskBase 一键启动（Windows）
# 用法：双击 start.bat，或在 PowerShell 里执行 .\start.ps1 -Port 8010
param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Fail($message) {
    Write-Host ""
    Write-Host "[X] $message" -ForegroundColor Red
    Write-Host ""
    Read-Host "按回车键退出"
    exit 1
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AskBase - 企业知识库问答" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Fail "没有找到 python。请安装 Python 3.11+ 并勾选 'Add Python to PATH'。下载：https://www.python.org/downloads/"
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# ---------- 1. 环境 ----------
if (-not (Test-Path $venvPython)) {
    Write-Host "[1/3] 首次运行，创建虚拟环境..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "创建虚拟环境失败。" }

    Write-Host "[2/3] 安装依赖，第一次需要几分钟..." -ForegroundColor Yellow
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "可以试试国内镜像：" -ForegroundColor Yellow
        Write-Host "  .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
        Fail "安装依赖失败。"
    }
} else {
    Write-Host "[1/3] 虚拟环境已就绪" -ForegroundColor Green
}

# ---------- 2. 知识库 ----------
if (-not (Test-Path "data\app.db")) {
    Write-Host "[2/3] 导入示例文档..." -ForegroundColor Yellow
    & $venvPython scripts\seed.py
    if ($LASTEXITCODE -ne 0) { Fail "导入示例文档失败。" }
} else {
    Write-Host "[2/3] 知识库已就绪" -ForegroundColor Green
}

# ---------- 3. 配置提示 ----------
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "[!] 没有找到 .env，将以 Mock 模式启动（答案由检索结果抽取生成）" -ForegroundColor Yellow
    Write-Host "    接入真实模型：复制 .env.example 为 .env，填入 LLM_API_KEY 后重新启动" -ForegroundColor Yellow
    Write-Host ""
}

# ---------- 4. 启动 ----------
$url = "http://127.0.0.1:$Port"
Write-Host "[3/3] 启动服务: $url" -ForegroundColor Green
Write-Host "      关闭本窗口或按 Ctrl+C 停止服务" -ForegroundColor Gray
Write-Host ""

if (-not $NoBrowser) {
    # 后台等 4 秒再开浏览器，避免服务还没起来就访问
    Start-Job -ScriptBlock {
        param($u)
        Start-Sleep -Seconds 4
        Start-Process $u
    } -ArgumentList $url | Out-Null
}

& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port $Port

Write-Host ""
Write-Host "服务已停止。" -ForegroundColor Gray
