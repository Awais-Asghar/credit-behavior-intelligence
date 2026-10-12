param(
    [Parameter(Mandatory = $false)]
    [string]$InputPath = "$(Split-Path -Parent $PSScriptRoot)\reports\presentation\Customer_Intelligence_ML_Pipeline.pptx",

    [Parameter(Mandatory = $false)]
    [string]$OutputPath = "$(Split-Path -Parent $PSScriptRoot)\reports\presentation\Customer_Intelligence_ML_Pipeline.pdf",

    [Parameter(Mandatory = $false)]
    [string]$PreviewDirectory = ""
)

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $resolvedOutput
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null

$powerPoint = $null
$presentation = $null

try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $powerPoint.Visible = 1
    $presentation = $powerPoint.Presentations.Open($resolvedInput, $true, $false, $false)
    $presentation.SaveAs($resolvedOutput, 32)

    if ($PreviewDirectory) {
        $resolvedPreview = [System.IO.Path]::GetFullPath($PreviewDirectory)
        [System.IO.Directory]::CreateDirectory($resolvedPreview) | Out-Null
        $presentation.Export($resolvedPreview, "PNG", 1920, 1080)
    }

    Write-Output "Exported PDF to $resolvedOutput"
}
finally {
    if ($null -ne $presentation) {
        $presentation.Close()
    }
    if ($null -ne $powerPoint) {
        $powerPoint.Quit()
    }
}
