# Srovnání rychlostních profilů na krátkých úsecích (49 snímků). Výsledky: output\srovnani_profilu\<klip>_<profil>
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = ".\.venv\Scripts\python.exe"
$out = "$root\output\srovnani_profilu"
New-Item -ItemType Directory -Force $out | Out-Null
$runs = @(
  @("vlasy_obloha_1080", "turbo"), @("vlasy_obloha_1080", "fast"), @("vlasy_obloha_1080", "normal"),
  @("muz_zed_1080", "turbo"), @("muz_zed_1080", "fast"), @("muz_zed_1080", "normal"),
  @("kudrny_720", "turbo"), @("kudrny_720", "normal")
)
foreach ($r in $runs) {
  $clip, $prof = $r
  $t0 = Get-Date
  $log = & $py -m alphagen_studio.cli run "tests\data\cmp\$clip.mp4" --profile $prof --formats matte_png --output "$out\${clip}_$prof" 2>&1 | Out-String
  $sec = [int]((Get-Date) - $t0).TotalSeconds
  $seg = ([regex]::Match($log, "Segment 1/1 \S+ \S+ ([\d\.]+) s")).Groups[1].Value
  $vram = ([regex]::Match($log, "([\d\.]+) GB\.")).Groups[1].Value
  $err = ([regex]::Match($log, "!! .*")).Value
  "$clip;$prof;$sec;$seg;$vram;$err" | Tee-Object -FilePath "$out\casy.csv" -Append
}
"HOTOVO"
