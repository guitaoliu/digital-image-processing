%%
img_base = imread('data/Image A.jpg');
img_rotated = imread('data/Image B.jpg');
subplot(1,2,1)
imshow(img_base)
title('参考图像')
subplot(2,2,2)
imshow(img_rotated)
title('原始图像')

cpselect(img_rotated, img_base);
%%
base_points = [1196.56250000000,1694.18750000000;2129.12500000000,1246.62500000000;2235.87500000000,2085.12500000000;1904.37500000000,734.375000000000;1053.62500000000,2260.12500000000;975.875000000000,774.125000000000;2981.62500000000,1286.87500000000];
rotated_points = [906.562500000000,1251.93750000000;1923.37500000000,1059.37500000000;1810.62500000000,1896.62500000000;1836.62500000000,505.375000000000;622.375000000000,1762.62500000000;929.875000000000,306.875000000000;2736.37500000000,1316.87500000000];

tform = fitgeotrans(rotated_points, base_points, 'affine');
img_out = imwarp(img_rotated, tform);
tform2 = fitgeotrans(base_points, rotated_points, 'affine');
img_rotated2 = imwarp(img_base, tform2);
subplot(2,2,1)
imshow(img_base)
title('参考图像')
subplot(2,2,2)
imshow(img_out)
title('配准图像')
subplot(2,2,3)
imshow(img_rotated)
title('参考图像')
subplot(2,2,4)
imshow(img_rotated2)
title('配准图像')

%%
imwrite(img_out, 'result/matlab/img_out.jpg')
imwrite(img_rotated2, 'result/matlab/img_rotated.jpg')
