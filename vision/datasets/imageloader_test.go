package datasets

import (
	"log"
	"testing"
	"time"

	"github.com/wangkuiyi/gotorch/vision/transforms"
)

/*
func TestImageTgzLoader(t *testing.T) {
	a := assert.New(t)
	d, e := ioutil.TempDir("", "gotorch_image_tgz_loader*")
	a.NoError(e)

	fn := tgz.SynthesizeTarball(t, d)
	expectedVocab := map[string]int64{"0": int64(0), "1": int64(1)}
	vocab, e := BuildLabelVocabularyFromTgz(fn)
	a.NoError(e)
	a.Equal(expectedVocab, vocab)

	trans := transforms.Compose(
		transforms.ToTensor(),
		transforms.Normalize([]float64{0.1307}, []float64{0.3081}),
	)
	loader, e := NewImageLoader(fn, vocab, trans, 3)
	a.NoError(e)
	{
		// first iteration
		a.True(loader.Scan())
		data, label := loader.Minibatch()
		a.Equal([]int64{3, 3, 2, 2}, data.Shape())
		a.Equal([]int64{3}, label.Shape())
	}
	{
		// second iteration with minibatch size is 2
		a.True(loader.Scan())
		data, label := loader.Minibatch()
		a.Equal([]int64{2, 3, 2, 2}, data.Shape())
		a.Equal([]int64{2}, label.Shape())
	}
	// no more data at the third iteration
	a.False(loader.Scan())
	a.NoError(loader.Err())

	_, e = BuildLabelVocabularyFromTgz("no file")
	a.Error(e)
}
*/
func TestImageTgzLoaderHeavy(t *testing.T) {
	trainFn := "/Users/yancey/.cache/imagenet/imagenet_train_shuffle_1k.tgz"
	mbSize := 32
	vocab, e := BuildLabelVocabularyFromTgz(trainFn)
	if e != nil {
		log.Fatal(e)
	}
	trans := transforms.Compose(
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(0.5),
		transforms.Normalize([]float32{0.485, 0.456, 0.406}, []float32{0.229, 0.224, 0.225}))

	loader, e := NewImageLoader(trainFn, vocab, trans, mbSize)
	if e != nil {
		log.Fatal(e)
	}
	startTime := time.Now()
	idx := 0
	for loader.Scan() {
		//torch.GC()
		idx++
		loader.Minibatch()
		if idx%10 == 0 {
			throughput := float64(mbSize*10) / time.Since(startTime).Seconds()
			log.Printf("throughput: %f samples/secs", throughput)
			startTime = time.Now()
		}
	}
}
