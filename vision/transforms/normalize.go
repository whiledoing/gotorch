package transforms

import (
	"image"
)

const denom = float32(0xffff)

// ImageFloat struct
type ImageFloat struct {
	Array []float32
	Shape []int
}

// NormalizeTransformer corresponds to torchvision.transforms.html#Normalize. It
// implements Go interface gotorch/data.Transform.
type NormalizeTransformer struct {
	Mean, Stddev []float32
}

// Normalize returns normalize transformer
func Normalize(mean []float32, stddev []float32) *NormalizeTransformer {
	/**
		var meanArray []float32
		var stddevArray []float32
		if len(mean) == 1 {
			meanArray = []float32{mean[0], mean[0], mean[0]}
		} else if len(mean) == 3 {
			meanArray = mean
		} else {
			panic(fmt.Sprintf("len(Mean) should be 1 or 3."))
		}
		if len(stddev) == 1 {
			stddevArray = []float32{stddev[0], stddev[0], stddev[0]}
		} else if len(stddev) == 3 {
			stddevArray = stddev
		} else {
			panic(fmt.Sprintf("len(Stddev) should be 1 or 3."))
		}
	**/
	return &NormalizeTransformer{mean, stddev}
}

// Run normalize the input (Tensor) of size (C, H, W) using the stats value mean, stddev.
func (t *NormalizeTransformer) Run(img image.Image) ImageFloat {
	switch img.(type) {
	case *image.Gray, *image.Gray16:
		return t.normalizeGrayImage(img)
	}
	return t.normalizeColorImage(img)
}

func (t *NormalizeTransformer) normalizeColorImage(img image.Image) ImageFloat {
	maxX, maxY := img.Bounds().Max.X, img.Bounds().Max.Y
	array := make([]float32, maxY*maxX*3) // 1 channels
	i := 0
	for y := 0; y < maxY; y++ {
		for x := 0; x < maxX; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			array[i] = (float32(r)/denom - t.Mean[0]) / t.Stddev[0]
			i++
			array[i] = (float32(g)/denom - t.Mean[1]) / t.Stddev[1]
			i++
			array[i] = (float32(b)/denom - t.Mean[2]) / t.Stddev[2]
			i++
		}
	}
	return ImageFloat{
		Array: array,
		Shape: []int{maxY, maxX, 3},
	}
}

func (t *NormalizeTransformer) normalizeGrayImage(img image.Image) ImageFloat {
	maxX, maxY := img.Bounds().Max.X, img.Bounds().Max.Y
	array := make([]float32, maxY*maxX) // 1 channels
	i := 0
	for y := 0; y < maxY; y++ {
		for x := 0; x < maxX; x++ {
			r, _, _, _ := img.At(x, y).RGBA()
			array[i] = (float32(r)/denom - t.Mean[0]) / t.Stddev[0]
			i++
		}
	}
	return ImageFloat{
		Array: array,
		Shape: []int{maxY, maxX, 1},
	}
}
