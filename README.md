# octopus_simulation

## Install Python

A convenience PowerShell script is included to install Python on Windows. It will try to use `winget` first, then fall back to the bundled MSIX (`python-manager-26.2.msix`) if present.

Run PowerShell as Administrator and execute:

```powershell
.\install-python.ps1
```

If `winget` is not available and the MSIX is not present, download Python from https://www.python.org/downloads/ and install manually.
