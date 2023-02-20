package main

import (
	"image/color"
	"math"

	"github.com/solarlune/tetra3d"
	"github.com/solarlune/tetra3d/colors"
	"github.com/solarlune/tetra3d/examples"

	"github.com/hajimehoshi/ebiten/v2"
)

//const w = 796
//const h = 448

const w = 480 * 2
const h = 270 * 2

//const w = 1920
//const h = 1024

type Game struct {
	Library       *tetra3d.Library
	Width, Height int
	Scene         *tetra3d.Scene
	Camera        examples.BasicFreeCam
	SystemHandler examples.BasicSystemHandler
}

func NewGame() *Game {
	game := &Game{
		Width:  w,
		Height: h,
	}

	game.Init()

	return game
}

func NewSurfaceMesh(v []float32) *tetra3d.Mesh {
	w := 128
	h := 128

	mesh := tetra3d.NewMesh("Surface")

	vi := make([]tetra3d.VertexInfo, 0, 128*128)
	for j := 0; j < h; j++ {
		for i := 0; i < w; i++ {
			vert := tetra3d.NewVertex(float64(i)*0.1, math.Sin(float64(j*w+i)/10.0), float64(j)*0.1, 0, 0)
			vert.Colors = append(vert.Colors, tetra3d.NewColorFromHSV(0.35-float64(i+j)/(256.0)*0.3, 0.5, 0.7))
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

// In this example, we will simply create a cube and place it in the scene.

func (g *Game) Init() {
	g.Scene = tetra3d.NewScene("cubetest")

	g.Camera = examples.NewBasicFreeCam(g.Scene)
	g.Camera.Camera.Resize(w, h)
	//g.Camera.Camera.SetNear(0.1)
	//g.Camera.Camera.SetFar(16.0)
	//g.Camera.Camera.SetFieldOfView(90)
	//g.Camera.Move(0, 1, 1)
	g.Camera.Move(0, 20, 5)

	light := tetra3d.NewPointLight("light", 1, 1, 1, 2)
	light.Distance = 100
	light.Move(6, 10, 6)
	light.On = true
	g.Scene.Root.AddChildren(light)

	//cube := tetra3d.NewModel(tetra3d.NewCubeMesh(), "Cube")
	//cube.Move(6, 10, 6)
	//g.Scene.Root.AddChildren(cube)

	g.SystemHandler = examples.NewBasicSystemHandler(g)
	//g.Scene.World.LightingOn = false

	/*
		for i := 1; i <= 128; i++ {
			//c1 := tetra3d.NewModel(cube.Mesh.Clone(), "c1")
			c1 := cube.Clone().(*tetra3d.Model)
			g.Scene.Root.AddChildren(c1)
			for j := 1; j <= 128; j++ {
				// This is needed if StaticMerge is done
				//c2 := tetra3d.NewModel(cube.Mesh.Clone(), "c2")
				c2 := c1.Clone().(*tetra3d.Model)
				c2.Move(float64(i), float64(i+j), -float64(j))
				c1.DynamicBatchAdd(c1.Mesh.MeshParts[0], c2)
			}
		}
	*/

	v := make([]float32, 128*128)
	surf := tetra3d.NewModel(NewSurfaceMesh(v), "Surface")
	//surf.Color.Set(1, 0.5, 0, 0)
	g.Scene.Root.AddChildren(surf)

	//cube := tetra3d.NewModel(tetra3d.NewCubeMesh(), "Cube")
	//cube.Color.Set(0, 0.5, 1, 1)
	//g.Scene.Root.AddChildren(cube)

	// Create a camera, move it back.
	//g.Camera = tetra3d.NewCamera(g.Width, g.Height)
	//g.Camera = camera

	// Again, we don't need to actually add the camera to the scenegraph, but we'll do it anyway because why not.
	//g.Scene.Root.AddChildren(g.Camera)

}

func (g *Game) Update() error {

	// Spinning the cube.
	//cube := g.Scene.Root.Get("Cube")
	//cube.SetLocalRotation(cube.LocalRotation().Rotated(0, 1, 0, 0.05))

	g.Camera.Update()

	return g.SystemHandler.Update()
}

func (g *Game) Draw(screen *ebiten.Image) {

	// Clear the screen with a color.
	screen.Fill(color.RGBA{60, 70, 80, 255})

	// Clear the Camera.
	g.Camera.Clear()

	// Render the scene.
	g.Camera.RenderScene(g.Scene)

	// Draw depth texture if the debug option is enabled; draw color texture otherwise.
	screen.DrawImage(g.Camera.ColorTexture(), nil)

	g.SystemHandler.Draw(screen, g.Camera.Camera)

	if g.SystemHandler.DrawDebugText {
		txt := `F1 to toggle this text
This is a very simple example showing
a simple 3D cube, created through code.
F5: Toggle depth debug view
F4: Toggle fullscreen
ESC: Quit`
		g.Camera.DebugDrawText(screen, txt, 0, 200, 1, colors.LightGray())
	}
}

func (g *Game) Layout(w, h int) (int, int) {
	// This is a fixed aspect ratio; we can change this to, say, extend for wider displays by using the provided w argument and
	// calculating the height from the aspect ratio, then calling Camera.Resize() with the new width and height.
	return g.Width, g.Height
}

func main() {

	ebiten.SetWindowTitle("Tetra3d - Simple Test")

	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeEnabled)

	game := NewGame()

	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}
