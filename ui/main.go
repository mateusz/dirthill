package main

import (
	"encoding/json"
	"fmt"
	"image/color"
	"log"
	"math"
	"time"

	_ "embed"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"

	"syscall/js"

	"github.com/solarlune/tetra3d"
)

//go:embed grassblock.gltf
var grassBlockBytes []byte

const (
	screenWidth  = 640
	screenHeight = 480
	csWidth      = 128
	csHeight     = 64
	csHRatio     = 2
	csWRatio     = 4
)

var (
	neutralFill = color.RGBA{60, 70, 80, 255}
)

type Game struct {
	console       js.Value
	document      js.Value
	crossSection  *ebiten.Image
	csValues      []float32
	tileValues    []float32
	Library       *tetra3d.Library
	Scene         *tetra3d.Scene
	Camera        *tetra3d.Camera
	Cube          *tetra3d.Model
	lastChanged   time.Time
	changed       bool
	needs3dRender bool
}

func (g *Game) Init() {
	library, err := tetra3d.LoadGLTFData(grassBlockBytes, nil)
	if err != nil {
		panic(err)
	}

	g.Library = library
	g.Cube = library.Scenes[0].Root.SearchTree().ByName("Cube").Models()[0]
	//camera := library.Scenes[0].Root.SearchTree().ByName("Camera").First().(*tetra3d.Camera)

	g.Scene = tetra3d.NewScene("cubetest")
	g.Scene.World.LightingOn = false

	g.Camera = tetra3d.NewCamera(screenWidth, screenHeight-csHeight*csHRatio-10)
	//g.Camera.SetFar(128)
	g.Camera.Move(64, 30, 64)

	asyncWait := make(chan interface{})
	g.document.Call("load").Call("then", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		g.console.Call("log", "Model loaded")
		asyncWait <- nil
		return nil
	}))

	<-asyncWait
}

func (g *Game) Infer() {
	edge, err := json.Marshal(g.csValues)
	if err != nil {
		g.console.Call("log", fmt.Sprintf("Failed to marshal data: %s", err))
	}

	asyncWait := make(chan interface{})
	g.document.Call("infer", string(edge)).Call("then", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		for i := 0; i < 128*128; i++ {
			g.tileValues[i] = float32(args[0].Get(fmt.Sprintf("%d", i)).Float())
		}
		asyncWait <- nil
		return nil
	}))

	<-asyncWait
}

func (g *Game) Update() error {
	g.crossSection.Fill(color.White)

	mx, my := ebiten.CursorPosition()
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		csx := mx - 10
		csy := my - (screenHeight - csHeight*csHRatio - 10)
		if csx >= 0 && csx < csWidth*csWRatio && csy >= 0 && csy < csHeight*csHRatio {
			vx := int(math.Floor(float64(csx) / csWRatio))
			vy := float64(64 - csy/csHRatio)
			if g.csValues[vx] != float32(vy) {
				g.csValues[vx] = float32(vy)
				g.changed = true
				g.lastChanged = time.Now()
			}
		}
	}

	for i, v := range g.csValues {
		ebitenutil.DrawRect(
			g.crossSection,
			float64(i*csWRatio),
			float64((csHeight-v)*csHRatio),
			float64(csWRatio),
			float64(csHRatio),
			color.Black,
		)

	}

	if g.changed && time.Now().Sub(g.lastChanged) > 1*time.Second {
		g.changed = false

		g.Scene.Root.SearchTree().ForEach(func(node tetra3d.INode) bool {
			g.Scene.Root.RemoveChildren(node)
			return true
		})

		g.Infer()

		for k := 0; k < 4; k++ {
			c1 := g.Cube.Clone().(*tetra3d.Model)
			g.Scene.Root.AddChildren(c1)
			for j := 0; j < 4; j++ {
				for i := 0; i < 128; i++ {
					c2 := c1.Clone().(*tetra3d.Model)
					c2.Move(float64(i), float64(g.tileValues[(32*k+j)*128+i]), float64(j))
					c1.DynamicBatchAdd(c1.Mesh.MeshParts[0], c2)
				}
			}
		}
		g.needs3dRender = true
	}

	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(neutralFill)

	if g.needs3dRender {
		g.needs3dRender = false
		g.Camera.Clear()
		g.Camera.RenderScene(g.Scene)
	}
	screen.DrawImage(g.Camera.ColorTexture(), nil)

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(float64(10), float64(screenHeight-csHeight*csHRatio-10))
	screen.DrawImage(g.crossSection, op)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

func main() {
	g := &Game{
		crossSection: ebiten.NewImage(csWidth*csWRatio, csHeight*csHRatio),
		csValues:     make([]float32, csWidth),
		tileValues:   make([]float32, 128*128),
		console:      js.Global().Get("console"),
		document:     js.Global().Get("document"),
	}
	g.Init()

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetWindowTitle("Test")
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}
