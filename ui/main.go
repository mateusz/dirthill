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
	"github.com/hajimehoshi/ebiten/v2/text"

	"syscall/js"

	"github.com/solarlune/tetra3d"
	"golang.org/x/image/font/basicfont"
)

//go:embed grassblock.gltf
var grassBlockBytes []byte

const (
	screenWidth  = 640
	screenHeight = 480
	csWidth      = 128
	csHeight     = 50
	csHRatio     = 1
	csWRatio     = 2
)

var (
	neutralFill   = color.RGBA{60, 70, 80, 255}
	neutralBright = color.RGBA{78, 90, 102, 255}
	csHighlight1  = color.RGBA{251, 105, 80, 255}
	csHighlight2  = color.RGBA{47, 144, 212, 255}
	waterColor    = color.RGBA{50, 50, 212, 255}
	waterLevel    = 3.0
)

func htoc(h float64) *tetra3d.Color {
	return tetra3d.NewColorFromHSV(0.35-(h/60.0)*0.3, 0.5, 0.7)
}

type crossSection struct {
	highlight color.Color
	canvas    *ebiten.Image
	values    []float32
	x         float64
	y         float64
}

func newCrossSection(x, y float64, highlight color.Color) *crossSection {
	cs := &crossSection{
		highlight: highlight,
		canvas:    ebiten.NewImage(csWidth*csWRatio, csHeight*csHRatio),
		values:    make([]float32, csWidth),
		x:         x,
		y:         y,
	}
	for i := 0; i < 128; i++ {
		cs.values[i] = csHeight / 2.0
	}
	cs.canvasPaint()
	return cs
}

func (cs *crossSection) mouseEvent(mx, my int) bool {
	changed := false
	if ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft) {
		csx := mx - int(cs.x)
		csy := my - int(cs.y)
		if csx >= 0 && csx < csWidth*csWRatio && csy >= 0 && csy < csHeight*csHRatio {
			vx := int(math.Floor(float64(csx) / csWRatio))
			vy := float64(csHeight - csy/csHRatio)
			if cs.values[vx] != float32(vy) {
				cs.values[vx] = float32(vy)
				changed = true
			}
		}
	}

	if changed {
		cs.canvasPaint()
	}

	return changed
}

func (cs *crossSection) canvasPaint() {
	cs.canvas.Fill(neutralBright)
	for i, v := range cs.values {
		ebitenutil.DrawRect(
			cs.canvas,
			float64(i*csWRatio),
			float64((csHeight-v)*csHRatio),
			float64(csWRatio),
			float64(csHeight*csHRatio),
			htoc(float64(v)).ToRGBA64(),
		)

		ebitenutil.DrawRect(
			cs.canvas,
			float64(i*csWRatio),
			float64((csHeight-v)*csHRatio),
			float64(csWRatio),
			float64(csHRatio),
			cs.highlight,
		)

	}
}

func (cs *crossSection) draw(screen *ebiten.Image) {
	ebitenutil.DrawRect(
		screen,
		cs.x-1,
		cs.y-1,
		float64(csWidth*csWRatio)+2,
		float64(csHeight*csHRatio)+2,
		color.Black,
	)

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(cs.x, cs.y)
	screen.DrawImage(cs.canvas, op)
}

type Game struct {
	sides         int
	modelName     string
	console       js.Value
	document      js.Value
	cs1           *crossSection
	cs2           *crossSection
	tileValues    []float32
	Library       *tetra3d.Library
	Scene         *tetra3d.Scene
	Camera        *tetra3d.Camera
	Cube          *tetra3d.Model
	lastChanged   time.Time
	changed       bool
	needs3dRender bool
}

func (g *Game) NewSurfaceMesh(w, h int) *tetra3d.Mesh {
	mesh := tetra3d.NewMesh("Surface")

	vi := make([]tetra3d.VertexInfo, 0, 128*128)
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			// values generally between 0..csHeight (but could be more or less in extreme cases if model returns values outside of training range)
			elev := float64(g.tileValues[(127-j)*w+i])
			vert := tetra3d.NewVertex(float64(j)*0.1, elev*0.15, float64(i)*0.1, 0, 0)
			if elev < waterLevel {
				vert = tetra3d.NewVertex(float64(j)*0.1, waterLevel*0.15, float64(i)*0.1, 0, 0)
				vert.Colors = append(vert.Colors, tetra3d.NewColor(float32(waterColor.R)/255.0, float32(waterColor.G)/255.0, float32(waterColor.B)/255.0, float32(waterColor.A)/255.0))
			} else if i <= 1 {
				vert.Colors = append(vert.Colors, tetra3d.NewColor(float32(csHighlight1.R)/255.0, float32(csHighlight1.G)/255.0, float32(csHighlight1.B)/255.0, float32(csHighlight1.A)/255.0))
			} else if g.sides > 1 && j <= 1 {
				vert.Colors = append(vert.Colors, tetra3d.NewColor(float32(csHighlight2.R)/255.0, float32(csHighlight2.G)/255.0, float32(csHighlight2.B)/255.0, float32(csHighlight2.A)/255.0))
			} else {
				vert.Colors = append(vert.Colors, htoc(elev*1.2))
			}

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

			// Front face
			tris = append(tris, j0+i)
			tris = append(tris, j1+i)
			tris = append(tris, j0+i+1)
			tris = append(tris, j0+i+1)
			tris = append(tris, j1+i)
			tris = append(tris, j1+i+1)

			// Back face
			tris = append(tris, j0+i)
			tris = append(tris, j0+i+1)
			tris = append(tris, j1+i)
			tris = append(tris, j0+i+1)
			tris = append(tris, j1+i+1)
			tris = append(tris, j1+i)

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

	g.Camera = tetra3d.NewCamera(screenWidth, screenHeight)
	//g.Camera.SetFar(128)
	g.Camera.Move(-5, 10, -5)
	g.Camera.Rotate(0, 1, 0, -0.75*math.Pi)
	g.Camera.Rotate(1, 0, 0, -0.5)
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

	surf := tetra3d.NewModel(g.NewSurfaceMesh(128, 128), "Surface")
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

func (g *Game) Interp() {
	if g.sides == 1 {
		g.console.Call("log", "Cannot interpolate 1 edge")
		return
	}

	for j := 0; j < 128; j++ {
		for i := 0; i < 128; i++ {
			g.tileValues[j*128+i] = (g.cs1.values[j] * g.cs2.values[i]) / 60.0
		}
	}
}

func (g *Game) Infer() {
	var edge []byte
	var err error
	var values []float32
	if g.sides == 1 {
		values = make([]float32, 0, 128)
		values = append(values, g.cs1.values...)
	} else {
		values = make([]float32, 0, 256)
		values = append(values, g.cs1.values...)
		values = append(values, g.cs2.values...)
	}

	min := float32(99999.0)
	max := float32(0.0)
	for i, _ := range values {
		if values[i] < min {
			min = values[i]
		}
		if values[i] > max {
			max = values[i]
		}
	}

	// Discard data below and above minimum (model expect normalised [-1,1] input)
	span := max - min
	for i, _ := range values {
		values[i] = (values[i]-min)/(span/2.0) - 1.0
	}

	edge, err = json.Marshal(values)

	if err != nil {
		g.console.Call("log", fmt.Sprintf("Failed to marshal data: %s", err))
	}

	asyncWait := make(chan interface{})
	g.document.Call("infer", string(edge)).Call("then", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		obase := float32(99999.0)
		for i := 0; i < 128*128; i++ {
			v := float32(args[0].Get(fmt.Sprintf("%d", i)).Float())
			if v < obase {
				obase = v
			}

		}

		for i := 0; i < 128*128; i++ {
			v := float32(args[0].Get(fmt.Sprintf("%d", i)).Float())
			// Pull output down to zero, then rescale back to reflect the input shape (which depends on csHeight).
			// However, let's always start at 0 - so don't push back up to min
			g.tileValues[i] = (v - obase) * (span / 2.0) // +min
		}
		asyncWait <- nil
		return nil
	}))

	<-asyncWait
}

func (g *Game) Update() error {
	mx, my := ebiten.CursorPosition()
	if g.cs1.mouseEvent(mx, my) {
		g.changed = true
		g.lastChanged = time.Now()

	}
	if g.sides > 1 && g.cs2.mouseEvent(mx, my) {
		g.changed = true
		g.lastChanged = time.Now()
	}

	if g.changed && time.Now().Sub(g.lastChanged) > 200*time.Millisecond {
		g.changed = false

		g.Scene.Root.SearchTree().ByName("Surface").ForEach(func(node tetra3d.INode) bool {
			g.Scene.Root.RemoveChildren(node)
			return true
		})

		g.Infer()

		surf := tetra3d.NewModel(g.NewSurfaceMesh(128, 128), "Surface")
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

	g.cs1.draw(screen)
	if g.sides > 1 {
		g.cs2.draw(screen)
	}

	opt := &ebiten.DrawImageOptions{}
	opt.GeoM.Translate(10, 23)
	text.DrawWithOptions(screen, fmt.Sprintf("Inferring terrain from cross-sections\nby github.com/mateusz\nModel: %s", g.modelName), basicfont.Face7x13, opt)
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

func main() {
	g := &Game{
		cs1:        newCrossSection(10, float64(screenHeight-csHeight*csHRatio-10), csHighlight1),
		cs2:        newCrossSection(screenWidth-csWidth*csWRatio-10, float64(screenHeight-csHeight*csHRatio-10), csHighlight2),
		tileValues: make([]float32, 128*128),
		console:    js.Global().Get("console"),
		document:   js.Global().Get("document"),
	}
	boundl := g.document.Get("boundl").Int()
	if boundl == 128 {
		g.sides = 1
	} else if boundl == 256 {
		g.sides = 2
	} else {
		fmt.Errorf("Bad boundl '%d', should be 128 or 256", boundl)
	}
	g.modelName = g.document.Get("modelName").String()

	g.Init()

	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)
	ebiten.SetWindowTitle("Test")
	if err := ebiten.RunGame(g); err != nil {
		log.Fatal(err)
	}
}
