% 加载MAT文件
matFile = 'D:\pythonprojects\fourcastnet21\outputs_fourcastnet_finetune\result\2023_5_20\input_data.mat';
load(matFile, 'data');  % 确保MAT文件中包含 'data' 变量

% 从 cell 数组中提取结构体
structData = data{1};  % 提取结构体
targetMatrix = structData.input;  % 直接从结构体的 input 字段中提取矩阵数据

% 提取二维切片
targetMatrix2D = squeeze(targetMatrix(1, 2, :, :));  % 提取二维切片并去除多余维度

% 经纬度范围
cols = size(targetMatrix2D, 2);
rows = size(targetMatrix2D, 1);
lon_range = linspace(115.0, 121.0, cols);  % 经度范围
lat_range = linspace(22.63, 19.0, rows);  % 纬度范围（反转，因MATLAB从上到下为增序）

% 可视化提取的数据
figure;
pcolor(lon_range, lat_range, targetMatrix2D);  % 使用pcolor确保经纬度比例
shading interp;  % 消除网格线效果
colormap(redblue3);  % 使用红蓝色映射
colorbar;
title('Input');
xlabel('Longitude (°E)');
ylabel('Latitude (°N)');

% 设置颜色范围
caxis([-1, 1]);  % 假设数据范围在 [-1, 1]，可以根据需要调整

% 设置轴的显示范围和刻度
xlim([115, 121]);
ylim([19, 22.63]);
set(gca, 'YDir', 'normal');  % 使纬度方向从下到上显示
xticks(115:1:121);  % 设置经度刻度
yticks(19:0.5:22.5);  % 设置纬度刻度
xticklabels({'115°E', '116°E', '117°E', '118°E', '119°E', '120°E', '121°E'});  % 经度标签
yticklabels({'19°N', '19.5°N', '20°N', '20.5°N', '21°N', '21.5°N', '22°N', '22.5°N'});  % 纬度标签

% 保存图像
outputFolder = 'D:\pythonprojects\fourcastnet21\outputs_fourcastnet_finetune\result\2023_5_20\';
saveas(gcf, fullfile(outputFolder, 'extracted_target_matrix_visualization.png'));
