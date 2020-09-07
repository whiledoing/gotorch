package datasets

import (
	"image"
	"io"
	"path/filepath"
	"unsafe"

	torch "github.com/wangkuiyi/gotorch"
	tgz "github.com/wangkuiyi/gotorch/tool/tgz"
	"github.com/wangkuiyi/gotorch/vision/transforms"
)

type sample struct {
	img   transforms.ImageFloat
	label int
}

// ImageLoader struct
type ImageLoader struct {
	r       *tgz.Reader
	vocab   map[string]int
	samples chan sample
	err     chan error
	trans   *transforms.ComposeTransformer
	mbSize  int
}

// NewImageLoader returns an ImageLoader
func NewImageLoader(fn string, vocab map[string]int, trans *transforms.ComposeTransformer, mbSize int) (*ImageLoader, error) {
	r, e := tgz.OpenFile(fn)
	if e != nil {
		return nil, e
	}
	m := &ImageLoader{
		r:       r,
		vocab:   vocab,
		samples: make(chan sample, mbSize*4),
		err:     make(chan error),
		trans:   trans,
		mbSize:  mbSize,
	}
	go m.retreiveMinibatch()
	return m, nil
}

// Scan return false if no more dat
func (p *ImageLoader) Scan() bool {
	select {
	case e := <-p.err:
		if e != nil {
			return false
		}
	default:
		return true
	}
	return true
}

func (p *ImageLoader) retreiveMinibatch() {
	defer close(p.samples)
	defer close(p.err)
	for {
		hdr, err := p.r.Next()
		if err != nil {
			p.err <- err
			break
		}
		if !hdr.FileInfo().Mode().IsRegular() {
			continue
		}
		classStr := filepath.Base(filepath.Dir(hdr.Name))
		label := p.vocab[classStr]

		m, _, err := image.Decode(p.r)
		if err != nil {
			p.err <- err
			break
		}
		input := p.trans.Run(m)
		p.samples <- sample{input.(transforms.ImageFloat), label}
	}
}

// Minibatch returns a minibash with data and label Tensor
func (p *ImageLoader) Minibatch() (torch.Tensor, torch.Tensor) {
	images := []torch.Tensor{}
	labels := []int64{}
	for i := 0; i < p.mbSize; i++ {
		sample, ok := <-p.samples
		if ok {
			tensorSize := []int64{}
			for _, v := range sample.img.Shape {
				tensorSize = append(tensorSize, int64(v))
			}
			t := torch.FromBlob(unsafe.Pointer(&sample.img.Array[0]), torch.Float, tensorSize)
			images = append(images, t)
			labels = append(labels, int64(sample.label))
		} else {
			break
		}
	}
	return torch.Stack(images, 0), torch.NewTensor(labels)
}

// Err returns the error during the scan process, if there is any. io.EOF is not
// considered an error.
func (p *ImageLoader) Err() error {
	if e, ok := <-p.err; ok && e != nil && e != io.EOF {
		return e
	}
	return nil
}

// BuildLabelVocabularyFromTgz build a label vocabulary from the image tgz file
func BuildLabelVocabularyFromTgz(fn string) (map[string]int, error) {
	vocab := make(map[string]int)
	l, e := tgz.ListFile(fn)
	if e != nil {
		return nil, e
	}
	idx := 0
	for _, hdr := range l {
		class := filepath.Base(filepath.Dir(hdr.Name))
		if _, ok := vocab[class]; !ok {
			vocab[class] = idx
			idx++
		}
	}
	return vocab, nil
}
