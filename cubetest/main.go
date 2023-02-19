package main

import (
	"image/color"

	_ "embed"

	"github.com/solarlune/tetra3d"
	"github.com/solarlune/tetra3d/colors"
	"github.com/solarlune/tetra3d/examples"

	"github.com/hajimehoshi/ebiten/v2"
)

//go:embed grassblock.gltf
var grassBlockBytes []byte

type Game struct {
	Library       *tetra3d.Library
	Width, Height int
	Scene         *tetra3d.Scene
	Camera        *tetra3d.Camera
	SystemHandler examples.BasicSystemHandler
}

func NewGame() *Game {
	game := &Game{
		Width:  796,
		Height: 448,
	}

	game.Init()

	return game
}

// In this example, we will simply create a cube and place it in the scene.

func (g *Game) Init() {
	library, err := tetra3d.LoadGLTFData(grassBlockBytes, nil)
	if err != nil {
		panic(err)
	}

	g.Library = library
	cube := library.Scenes[0].Root.SearchTree().ByName("Cube").Models()[0]
	camera := library.Scenes[0].Root.SearchTree().ByName("Camera").First().(*tetra3d.Camera)

	// Create a new Scene and name it.
	g.Scene = tetra3d.NewScene("cube example")
	g.Scene.World.LightingOn = false

	g.SystemHandler = examples.NewBasicSystemHandler(g)

	for i := 1; i <= 128; i++ {
		c1 := tetra3d.NewModel(cube.Mesh.Clone(), "c1")
		g.Scene.Root.AddChildren(c1)
		for j := 1; j <= 128; j++ {
			//c2 := tetra3d.NewModel(cube.Mesh.Clone(), "c2")
			c2 := c1.Clone()
			c2.Move(float64(i)*0.1, 0.0, -float64(j)*0.1)
			c1.DynamicBatchAdd(c1.Mesh.MeshParts[0], c2.(*tetra3d.Model))
		}
	}

	// Create a camera, move it back.
	//g.Camera = tetra3d.NewCamera(g.Width, g.Height)
	g.Camera = camera

	// Again, we don't need to actually add the camera to the scenegraph, but we'll do it anyway because why not.
	g.Scene.Root.AddChildren(g.Camera)

}

func (g *Game) Update() error {

	// Spinning the cube.
	//cube := g.Scene.Root.Get("Cube")
	//cube.SetLocalRotation(cube.LocalRotation().Rotated(0, 1, 0, 0.05))

	// Debug views

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

	g.SystemHandler.Draw(screen, g.Camera)

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
