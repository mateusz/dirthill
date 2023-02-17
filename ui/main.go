// https://github.com/golang/go/wiki/WebAssembly#getting-started
// https://www.aaron-powell.com/posts/2019-02-06-golang-wasm-3-interacting-with-js-from-go/
// https://github.com/microsoft/onnxruntime-inference-examples/blob/main/js/quick-start_onnxruntime-web-script-tag/index.html
// https://onnxruntime.ai/docs/build/web.html#prepare-onnx-runtime-webassembly-artifacts
package main

import (
	"encoding/json"
	"fmt"
	"syscall/js"
	"time"
)

func main() {
	console := js.Global().Get("console")

	d := make([]float32, 128)
	for i := range d {
		d[i] = 10.0
	}

	edge, err := json.Marshal(d)
	if err != nil {
		console.Call("log", "test")
	}

	document := js.Global().Get("document")
	asyncWait := make(chan interface{})

	for i := 1; i <= 10; i++ {
		start := time.Now()
		document.Call("infer", string(edge)).Call("then", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
			console.Call("log", js.ValueOf(args[0]))
			asyncWait <- nil
			return nil
		}))

		<-asyncWait
		console.Call("log", fmt.Sprintf("%.2fs", time.Now().Sub(start).Seconds()))
	}
}
