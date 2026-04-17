# AstralWeaver

一个 PySide6 全屏轨道屏保模拟器。

当前版本：`v2.2.1`

## 运行

```powershell
pip install -r requirements.txt
python .\astral_weaver.py
```

旧版桌面窗口入口仍保留在 `three_body_pyside.py`。

## 打包

生成 Windows 单文件版本：

```powershell
python -m PyInstaller --noconfirm --clean --onefile --windowed --name AstralWeaver-v2.2.1 --icon .\assets\three_body_v2_icon.ico --version-file version_info.txt astral_weaver.py
```

打包完成后入口为：

```powershell
.\dist\AstralWeaver-v2.2.1.exe
```

## 使用

- 启动后会进入全屏模拟；按 `Esc` 缩小为普通窗口，按 `F11` 切换全屏，双击画面回到全屏，按 `Q` 退出。
- 鼠标移到屏幕左侧可以展开设置面板，移开后面板会自动收起。
- 在设置面板中可以调整恒星数量；每颗恒星都可以通过标题旁的色块选择预设星光色，并设置位置、速度角度、速度大小和质量。
- 双击参数名可以锁定/解锁该参数；锁定后，单颗随机和总随机都会保留这个参数不变。
- 单颗恒星的随机按钮只改变该恒星参数，不会立刻重启模拟；点击 `应用 / 重启` 后才会使用当前参数重新开始。
- `随机刷新` 会保留当前恒星数量，并立即生成一组新的随机参数开始模拟。
- 预设菜单可选择 `经典 8 字`、`拉格朗日三角` 和 `欧拉共线`；也可以保存当前参数为自定义预设，并在之后修改或删除。
- 碰撞判定默认关闭；需要时双击 `碰撞` 参数名启用，再用碰撞距离控制判定范围。
- 模拟控制里可以调整步长、每帧计算步数、尾迹长度和交互距离；`恢复模拟默认值` 会把这些参数恢复到默认设置。
- 模拟结束后画面会淡出，并自动开始下一组随机运动。
