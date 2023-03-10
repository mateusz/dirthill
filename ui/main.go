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
	"github.com/hajimehoshi/ebiten/v2/inpututil"
	"github.com/hajimehoshi/ebiten/v2/text"

	"syscall/js"

	"github.com/solarlune/tetra3d"
	"golang.org/x/image/font/basicfont"
)

const (
	screenWidth  = 640
	screenHeight = 480
	csWidth      = 128
	csHeight     = 50
	csHRatio     = 1
	csWRatio     = 2
	cs3dAdjust   = 0.75
)

var (
	neutralFill   = color.RGBA{60, 70, 80, 255}
	neutralBright = color.RGBA{78, 90, 102, 255}
	csHighlight1  = color.RGBA{251, 105, 80, 255}
	csHighlight2  = color.RGBA{47, 144, 212, 255}
	waterColor    = color.RGBA{50, 50, 212, 255}
	waterLevel    = -100.0
)

func htoc(h float64) *tetra3d.Color {
	return tetra3d.NewColorFromHSV(0.35-(h/60.0)*0.3, 0.5, 0.7)
}

type crossSection struct {
	highlight color.Color
	canvas    *ebiten.Image
	values    []float32
	held      bool
	x         float64
	y         float64
	px        int
	py        int
}

func newCrossSection(x, y float64, highlight color.Color, title string) *crossSection {
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
	cs.canvas.Fill(neutralBright)
	cs.canvasPaint()

	opt := &ebiten.DrawImageOptions{}
	opt.GeoM.Translate(5, 15)
	text.DrawWithOptions(cs.canvas, fmt.Sprintf("Draw %s cross-section here", title), basicfont.Face7x13, opt)

	return cs
}

func (cs *crossSection) getExtremes(prevMin, prevMax float32) (min, max float32) {
	min = prevMin
	max = prevMax
	for i, _ := range cs.values {
		if cs.values[i] < min {
			min = cs.values[i]
		}
		if cs.values[i] > max {
			max = cs.values[i]
		}
	}
	return
}

func (cs *crossSection) placePixel(mx, my int) bool {
	changed := false
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

	return changed
}

func (cs *crossSection) mouseEvent(mx, my int, buttonPressed bool) bool {
	changed := false
	if buttonPressed {
		if cs.placePixel(mx, my) {
			changed = true
		}
	}
	if buttonPressed && cs.held && cs.px != 0 && cs.py != 0 {
		var xstep int
		if mx > cs.px {
			xstep = 1
		} else {
			xstep = -1
		}

		dy := float64(my - cs.py)
		for x := cs.px; x != mx; x += xstep {
			progress := float64(x-cs.px) / float64(mx-cs.px)
			if xstep < 0 {
				progress = progress
			}

			if cs.placePixel(x, cs.py+int(dy*progress)) {
				changed = true
			}
		}
	}

	if buttonPressed {
		cs.held = true
	} else {
		cs.held = false
	}

	if changed {
		cs.canvasPaint()
	}

	cs.px = mx
	cs.py = my
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

type game struct {
	mobileExperimental bool
	touchIDs           []ebiten.TouchID
	sides              int
	modelName          string
	console            js.Value
	document           js.Value
	cs1                *crossSection
	cs2                *crossSection
	tileValues         []float32
	library            *tetra3d.Library
	scene              *tetra3d.Scene
	camera             *tetra3d.Camera
	cameraAngle        float64
	cube               *tetra3d.Model
	lastChanged        time.Time
	changed            bool
	needs3dRender      bool
}

func (g *game) newSurfaceMesh(tileValues []float32, w, h int, colorOverride *color.RGBA) *tetra3d.Mesh {
	mesh := tetra3d.NewMesh("Surface")

	vi := make([]tetra3d.VertexInfo, 0, w*h)
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			// values generally between 0..csHeight (but could be more or less in extreme cases if model returns values outside of training range)
			elev := float64(tileValues[(h-j-1)*w+i])
			vert := tetra3d.NewVertex(float64(j)*0.1, elev*0.15, float64(i)*0.1, 0, 0)
			if elev < waterLevel {
				vert = tetra3d.NewVertex(float64(j)*0.1, waterLevel*0.15, float64(i)*0.1, 0, 0)
				vert.Colors = append(vert.Colors, tetra3d.NewColor(float32(waterColor.R)/255.0, float32(waterColor.G)/255.0, float32(waterColor.B)/255.0, float32(waterColor.A)/255.0))
			} else if colorOverride != nil {
				vert.Colors = append(vert.Colors, tetra3d.NewColor(float32(colorOverride.R)/255.0, float32(colorOverride.G)/255.0, float32(colorOverride.B)/255.0, float32(colorOverride.A)/255.0))
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

func (g *game) Init() {
	g.scene = tetra3d.NewScene("dirthill")
	g.scene.World.LightingOn = true

	g.camera = tetra3d.NewCamera(screenWidth, screenHeight)
	g.cameraAngle = -0.75

	light := tetra3d.NewPointLight("light", 1, 1, 1, 1.5)
	light.Distance = 100
	light.Move(-5, 20, -10)
	light.On = true
	g.scene.Root.AddChildren(light)
	g.updateSurfaces()

	asyncWait := make(chan interface{})
	g.document.Call("load").Call("then", js.FuncOf(func(this js.Value, args []js.Value) interface{} {
		g.console.Call("log", "Model loaded")
		asyncWait <- nil
		return nil
	}))

	<-asyncWait

	g.needs3dRender = true
}

func (g *game) interp() {
	if g.sides == 1 {
		g.console.Call("log", "Cannot interpolate 1 edge")
		return
	}

	for j := 0; j < 128; j++ {
		for i := 0; i < 128; i++ {
			g.tileValues[j*128+i] = (g.cs1.values[j] * g.cs2.values[i]) / 30.0
		}
	}
}

func (g *game) infer() {
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

	min, max := g.cs1.getExtremes(99999.0, 0.0)
	min, max = g.cs2.getExtremes(min, max)

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

func (g *game) Update() error {

	cooldown := 200 * time.Millisecond

	mx, my := ebiten.CursorPosition()
	g.touchIDs = ebiten.AppendTouchIDs(g.touchIDs[:0])
	if !g.mobileExperimental && mx > 0 && my > 0 {
		cooldown = 200 * time.Millisecond
		g.mobileExperimental = false

		pressed := ebiten.IsMouseButtonPressed(ebiten.MouseButtonLeft)
		if g.cs1.mouseEvent(mx, my, pressed) {
			g.changed = true
			g.lastChanged = time.Now()

		}
		if g.sides > 1 && g.cs2.mouseEvent(mx, my, pressed) {
			g.changed = true
			g.lastChanged = time.Now()
		}
	} else if len(g.touchIDs) > 0 {
		cooldown = 2000 * time.Millisecond
		g.mobileExperimental = true

		for _, id := range g.touchIDs {
			x, y := ebiten.TouchPosition(id)
			if x > 0 || y > 0 {
				if g.cs1.mouseEvent(x, y, true) {
					g.changed = true
					g.lastChanged = time.Now()
				}
				if g.sides > 1 && g.cs2.mouseEvent(x, y, true) {
					g.changed = true
					g.lastChanged = time.Now()
				}
			}
		}
	} else {
		g.cs1.mouseEvent(0, 0, false)
		g.cs2.mouseEvent(0, 0, false)
	}

	if g.changed && time.Now().Sub(g.lastChanged) > cooldown {
		g.changed = false
		g.infer()
		g.updateSurfaces()
	}

	if inpututil.IsKeyJustPressed(ebiten.KeyLeft) {
		g.cameraAngle += 0.1
		g.needs3dRender = true
	}
	if inpututil.IsKeyJustPressed(ebiten.KeyRight) {
		g.cameraAngle -= 0.1
		g.needs3dRender = true
	}

	return nil
}

func (g *game) updateSurfaces() {
	g.scene.Root.SearchTree().ByName("Surface").ForEach(func(node tetra3d.INode) bool {
		g.scene.Root.RemoveChildren(node)
		return true
	})

	g.scene.Root.AddChildren(tetra3d.NewModel(g.newSurfaceMesh(g.tileValues, 128, 128, nil), "Surface"))

	values := make([]float32, 0, 256)
	min, _ := g.cs1.getExtremes(99999.0, 0.0)
	min, _ = g.cs2.getExtremes(min, 0.0)

	for _, v := range g.cs1.values {
		n := (float32(v) - min) * cs3dAdjust
		values = append(values, n, n)
	}
	g.scene.Root.AddChildren(tetra3d.NewModel(g.newSurfaceMesh(values, 2, 128, &csHighlight1), "Surface"))

	if g.sides == 2 {
		for i, v := range g.cs2.values {
			n := (float32(v) - min) * cs3dAdjust
			values[i] = n
			values[128+i] = n
		}
		g.scene.Root.AddChildren(tetra3d.NewModel(g.newSurfaceMesh(values, 128, 2, &csHighlight2), "Surface"))
	}

	g.needs3dRender = true
}

func (g *game) Draw(screen *ebiten.Image) {
	screen.Fill(neutralFill)

	// There is a better way to do planetary rotation for camera, but for now this is fine.
	g.camera.ResetLocalTransform()
	g.camera.Move(0.1*64, 0, 0.1*64)
	g.camera.Move(17.0*math.Cos(g.cameraAngle*math.Pi), 10, 17.0*math.Sin(g.cameraAngle*math.Pi))
	g.camera.Rotate(0, 1, 0, -g.cameraAngle*math.Pi+math.Pi/2) // -0.75

	// Also, look down
	g.camera.Rotate(1, 0, 0, -0.5)

	if g.needs3dRender {
		g.needs3dRender = false
		g.camera.Clear()
		g.camera.RenderScene(g.scene)
	}
	screen.DrawImage(g.camera.ColorTexture(), nil)

	g.cs1.draw(screen)
	if g.sides > 1 {
		g.cs2.draw(screen)
	}

	opt := &ebiten.DrawImageOptions{}
	opt.GeoM.Translate(10, 23)
	text.DrawWithOptions(screen, fmt.Sprintf("Inferring terrain from cross-sections using deep learning\nSee https://github.com/mateusz/dirthill\nModel: %s\nUse arrows to rotate", g.modelName), basicfont.Face7x13, opt)

	if g.mobileExperimental {
		opt := &ebiten.DrawImageOptions{}
		opt.GeoM.Translate(screenWidth/2-120, 80)
		text.DrawWithOptions(screen, fmt.Sprintf("Caution: treat mobile as experimental"), basicfont.Face7x13, opt)
	}
}

func (g *game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return screenWidth, screenHeight
}

func main() {
	g := &game{
		cs1:        newCrossSection(10, float64(screenHeight-csHeight*csHRatio-10), csHighlight1, "red"),
		cs2:        newCrossSection(screenWidth-csWidth*csWRatio-10, float64(screenHeight-csHeight*csHRatio-10), csHighlight2, "blue"),
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
