gobuild:
	GOOS=js GOARCH=wasm go build -o dist/main.wasm

goinstallwasm:
	cp "$(go env GOROOT)/misc/wasm/wasm_exec.js" .