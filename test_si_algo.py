from si_algo_model import *
import os

def test_Downscale():
    scale = Downscale(0.5, 0.5)
    img = np.ones((10, 10))
    assert((scale(img) == np.ones((5, 5))).all())


def test_LR_HR_patch_pairs():
    img = np.eye(3, 3)
    img_ = np.eye(6, 6)
    for lpatch, hpatch in LR_HR_patch_pairs(img, img_):
        print(f"{lpatch=}, {hpatch=}")


def test_classify_patch_3x3():
    patch = np.ones((3, 3))
    print(classify_patch_3x3(patch))

def test_SuperInterpolation():
    si = SuperInterpolation()

    def dataloader():
        for i in range(10):
            yield np.random.randn(3, 3), np.random.randn(2, 2)


    si.fit(dataloader())
    LR_img = np.random.randn(256, 256)
    si.predict(LR_img)

    si.save("test.pkl")

    si2 = SuperInterpolation()
    si2.load("test.pkl")

    assert(si == si2)  # ensure load and save work
