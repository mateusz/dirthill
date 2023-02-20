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
	csHeight     = 32
	csHRatio     = 2
	csWRatio     = 2
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

func NewSurfaceMesh(w, h int, v []float32) *tetra3d.Mesh {
	//w := 128
	//h := 128

	mesh := tetra3d.NewMesh("Surface")

	vi := make([]tetra3d.VertexInfo, 0, 128*128)
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			elev := float64(v[j*w+i])
			vert := tetra3d.NewVertex(float64(i)*0.1, elev*0.1, float64(j)*0.1, 0, 0)
			vert.Colors = append(vert.Colors, tetra3d.NewColorFromHSV(0.35-(elev/40.0)*0.3, 0.5, 0.7))
			vert.ActiveColorChannel = 0
			vi = append(vi, vert)
		}
	}
	mesh.AddVertices(vi...)

	trisCounter := 0
	tris := make([]int, 0, (w-1)*(h-1)*12)
	for j := 0; j < h-1; j++ {
		for i := 0; i < w-1; i++ {
			j0 := j * w
			j1 := (j + 1) * w

			// Back face
			tris = append(tris, j0+i)
			tris = append(tris, j0+i+1)
			tris = append(tris, j1+i)
			tris = append(tris, j0+i+1)
			tris = append(tris, j1+i+1)
			tris = append(tris, j1+i)

			// Front face
			tris = append(tris, j0+i)
			tris = append(tris, j1+i)
			tris = append(tris, j0+i+1)
			tris = append(tris, j0+i+1)
			tris = append(tris, j1+i)
			tris = append(tris, j1+i+1)

			trisCounter += 4
			if trisCounter >= 21845-4 {
				trisCounter = 0
				mesh.AddMeshPart(tetra3d.NewMaterial("Surface"), tris...)
				tris = make([]int, 0, (w-1)*(h-1)*12)
			}
		}
	}

	if len(tris) > 0 {
		mesh.AddMeshPart(tetra3d.NewMaterial("Surface"), tris...)
	}

	mesh.UpdateBounds()
	mesh.AutoNormal()

	return mesh

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
	g.Scene.World.LightingOn = true

	g.Camera = tetra3d.NewCamera(screenWidth, screenHeight-csHeight*csHRatio-10)
	//g.Camera.SetFar(128)
	g.Camera.Move(-5, 10, -5)
	g.Camera.Rotate(0, 1, 0, -2.3)
	g.Camera.Rotate(1, 0, 0, -0.3)
	//g.Camera.Move(-100, 10, -100)

	light := tetra3d.NewPointLight("light", 1, 1, 1, 1.5)
	light.Distance = 100
	light.Move(-5, 20, -10)
	light.On = true
	g.Scene.Root.AddChildren(light)

	//cube := tetra3d.NewModel(tetra3d.NewCubeMesh(), "Cube")
	//cube.Move(6, 0, 6)
	//cube.Color.Set(0, 0.5, 1, 1)
	//g.Scene.Root.AddChildren(cube)

	surf := tetra3d.NewModel(NewSurfaceMesh(128, 128, g.tileValues), "Surface")
	g.Scene.Root.AddChildren(surf)

	asyncWait := make(chan interface{})
	g.document.Call("load").Call("then", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		g.console.Call("log", "Model loaded")
		asyncWait <- nil
		return nil
	}))

	<-asyncWait

	g.needs3dRender = true
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
		csx := mx - ((screenWidth / 2) - (csWidth * csWRatio / 2))
		csy := my - (screenHeight - csHeight*csHRatio - 10)
		if csx >= 0 && csx < csWidth*csWRatio && csy >= 0 && csy < csHeight*csHRatio {
			vx := int(math.Floor(float64(csx) / csWRatio))
			vy := float64(csHeight - csy/csHRatio)
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

	if g.changed && time.Now().Sub(g.lastChanged) > 200*time.Millisecond {
		g.changed = false

		g.Scene.Root.SearchTree().ByName("Surface").ForEach(func(node tetra3d.INode) bool {
			g.Scene.Root.RemoveChildren(node)
			return true
		})

		g.Infer()

		surf := tetra3d.NewModel(NewSurfaceMesh(128, 128, g.tileValues), "Surface")
		g.Scene.Root.AddChildren(surf)

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
	op.GeoM.Translate(float64(screenWidth)/2.0-(csWidth*csWRatio/2.0), float64(screenHeight-csHeight*csHRatio-10))
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
	for i := 1; i < 128; i++ {
		g.csValues[i] = 10.0
	}
	g.Init()

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetWindowTitle("Test")
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}
