% 加载MAT文件
matFile = 'D:\pythonprojects\fourcastnet21\outputs_fourcastnet_finetune\result\2024_4_9\48\true_data_U50.mat';
dataStruct = load(matFile);
data = dataStruct.data;

% 检查数据维度
[timeSteps, ~, rows, cols] = size(data);

% 经纬度范围
lon_range = linspace(115.0, 121.0, cols);  % 经度范围
lat_range = linspace(22.63, 19.0, rows);  % 纬度范围（反转，因MATLAB从上到下为增序）

% 创建保存图片的文件夹
outputFolder = 'D:\pythonprojects\fourcastnet21\outputs_fourcastnet_finetune\result\2024_4_9\48\true-U50\';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 设置颜色范围
colorLimits = [-1, 1];

% 设置GIF文件名
gifFile = fullfile(outputFolder, 'animated_visualization.gif');

% 循环遍历每个时间步并可视化
for t = 1:timeSteps
    % 创建一个新的图窗
    figure;

    % 提取当前时间步的数据，移除单例维度
    currentData = squeeze(data(t, 1, :, :));

    % 使用pcolor绘图，确保经纬度比例
    pcolor(lon_range, lat_range, currentData);
    shading interp;  % 消除网格线效果
    colormap(redblue3);  
    colorbar;
    caxis(colorLimits);  % 固定colorbar范围

    % 设置图形标题和坐标标签
    title(['Time Step: ', num2str(t)]);
    xlabel('Longitude');
    ylabel('Latitude');
    
    % 确保经纬度刻度等距
    axis equal;
    set(gca, 'YDir', 'normal');  % 使纬度方向从下到上显示

    % 调整轴的显示范围，消除上下空白区域
    xlim([115, 121]);
    ylim([19, 22.63]);

    % 保存当前图像
    saveas(gcf, fullfile(outputFolder, ['time_step_' num2str(t) '.png']));

    % 保存当前图像为临时文件
    frame = getframe(gcf);  % 获取当前图形的帧
    im = frame2im(frame);   % 将帧转换为图像

    % 确保是RGB格式
    [imind, cm] = rgb2ind(im, 256);  % 将RGB图像转换为索引图像
    
    % 将图像写入GIF文件
    if t == 1
        imwrite(imind, cm, gifFile, 'gif', 'LoopCount', inf, 'DelayTime', 1);  % 第一个图像
    else
        imwrite(imind, cm, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', 1);  % 后续图像
    end

    % 关闭图窗
    close;
end
