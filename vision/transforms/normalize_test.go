package transforms

import (
	"image"
	"image/color"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalizeTransformGrayImage(t *testing.T) {
	a := assert.New(t)

	trans := Normalize([]float32{10.0}, []float32{2.3})
	m := drawGrayImage(image.Rect(0, 0, 1, 1), color.Gray{127})
	r, _, _, _ := m.At(0, 0).RGBA()
	output := trans.Run(m)
	a.NoError(nil)
	expected := []float32{(float32(r)/denom - 10.0) / 2.3}
	a.Equal(output.Array, expected)
	a.Equal(output.Shape, []int{1, 1, 1})
}

func TestNormalizeTransformColorImage(t *testing.T) {
	a := assert.New(t)
	trans := Normalize([]float32{1.0, 2.0, 3.0}, []float32{2.3, 2.4, 2.5})
	m := drawImage(image.Rect(0, 0, 1, 1), blue)
	r, g, b, _ := m.At(0, 0).RGBA()
	// an image in torch should be a 3D tensor with CHW format
	output := trans.Run(m)
	expected := []float32{
		(float32(r)/denom - 1.0) / 2.3,
		(float32(g)/denom - 2.0) / 2.4,
		(float32(b)/denom - 3.0) / 2.5,
	}
	a.Equal(output.Array, expected)
	a.Equal(output.Shape, []int{1, 1, 3})
}

func TestNormalizeTransformPanic(t *testing.T) {
	a := assert.New(t)
	// mean and stddev should be 1 or 3 dims
	a.Panics(func() {
		Normalize([]float32{1.0, 2.0, 3.0, 4.0, 5.0}, []float32{2.3, 2.4, 2.5})
	})
	a.Panics(func() {
		Normalize([]float32{1.0, 2.0, 3.0}, []float32{2.3, 2.4})
	})

}
