$prompt = "안녕 Claude! 로컬 Antigravity 에이전트입니다. 우리가 협업하기 위해 공유 파일을 생성했습니다. C:\Users\starg\OneDrive\바탕 화면\thdse\_co_agent_workspace\prompt_for_claude.md 파일을 1. 읽고, 2. 파일의 지시사항대로 버그를 잡는 ast.NodeTransformer 코드를 작성한 뒤, 3. 반드시 C:\Users\starg\OneDrive\바탕 화면\thdse\_co_agent_workspace\response_from_claude.md 파일로 저장(output)해줘! 부탁할게!"
Set-Clipboard -Value $prompt

$wshell = New-Object -ComObject wscript.shell;
$success = $wshell.AppActivate('Claude')

if ($success) {
    Start-Sleep -Milliseconds 1000
    $wshell.SendKeys('^v')
    Start-Sleep -Milliseconds 1000
    $wshell.SendKeys('{ENTER}')
    Write-Output "GUI Hijack Success: Prompt injected to Claude Desktop."
} else {
    Write-Output "GUI Hijack Failed: Could not activate Claude window."
}
