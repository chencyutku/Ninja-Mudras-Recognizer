$PictureRoot = "${PSScriptRoot}\images"

Set-Location $PictureRoot
function rn($label)
{
    Get-ChildItem *.jpg | ForEach-Object -Begin {
        $i = 0
    } -Process {
        Rename-Item -Path $_ -NewName ("{0}_{1:d5}.jpg" -f $label, $i)
        $i++
    }
}

$list = Get-ChildItem

foreach ($dir in $list)
{
    Set-Location $dir
    rn $dir.BaseName
    Set-Location ..
}