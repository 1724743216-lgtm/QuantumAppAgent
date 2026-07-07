const { app, BrowserWindow, dialog } = require('electron');
const { fork, spawn, execSync } = require('child_process');
const path = require('path');
const net = require('net');
const fs = require('fs');
const tar = require('tar');

let mainWindow;
let splashWindow;
let serverProcess;
let backendProcess;
const PORT = 4716;
const BACKEND_PORT = 6174;

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 400,
    height: 300,
    frame: false,
    alwaysOnTop: true,
    transparent: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  // Simple HTML splash screen
  const splashHtml = `
    <!DOCTYPE html>
    <html>
    <body style="margin: 0; padding: 0; background: #1e1e1e; color: #fff; font-family: system-ui; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; border-radius: 8px;">
      <h2 style="margin-bottom: 10px;">TYQA 客户端</h2>
      <p id="status">正在初始化运行环境，请稍候...</p>
      <div style="width: 80%; height: 4px; background: #333; margin-top: 20px; border-radius: 2px; overflow: hidden;">
        <div style="width: 50%; height: 100%; background: #007acc; animation: indeterminate 1.5s infinite linear;"></div>
      </div>
      <style>
        @keyframes indeterminate {
          0% { transform: translateX(-100%); width: 50%; }
          100% { transform: translateX(200%); width: 50%; }
        }
      </style>
    </body>
    </html>
  `;
  splashWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(splashHtml)}`);
}

function ensurePythonEnv() {
  return new Promise(async (resolve, reject) => {
    if (!app.isPackaged) return resolve();
    const backendDir = path.join(process.resourcesPath, 'backend');
    const pythonDir = path.join(backendDir, 'backend_python');
    const tarPath = path.join(backendDir, 'backend_python.tar');

    if (fs.existsSync(pythonDir) || !fs.existsSync(tarPath)) {
      return resolve();
    }

    console.log('Extracting Python environment using npm tar...');
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.webContents.executeJavaScript(`document.getElementById('status').innerText = '首次启动，正在解压运行环境 (约需1-2分钟)，请耐心等待...'`);
    }

    try {
      await tar.x({
        file: tarPath,
        cwd: backendDir
      });
      resolve();
    } catch (err) {
      reject(new Error(`Node tar extraction failed: ${err.message}`));
    }
  });
}

function startBackend() {
  return new Promise(async (resolve, reject) => {
    try {
      await ensurePythonEnv();
    } catch (err) {
      dialog.showErrorBox('初始化失败', '无法解压运行环境，请尝试以管理员身份运行。\\n' + err.message);
      app.quit();
      return reject(err);
    }

    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.webContents.executeJavaScript(`document.getElementById('status').innerText = '正在启动核心服务...'`);
    }

    // If packaged, backend env is inside resources/backend/.venv
    const backendDir = app.isPackaged 
      ? path.join(process.resourcesPath, 'backend') 
      : path.join(__dirname, '..', '..'); // Points to QuantumAppAgent root in dev
    
    // Windows paths for the virtual environment
    const pythonExe = app.isPackaged
      ? path.join(backendDir, 'backend_python', 'python.exe')
      : 'uv'; // In dev (macOS/Linux), use uv

    const backendScriptsDir = app.isPackaged 
      ? path.join(backendDir, 'backend_python', 'Scripts')
      : '';
    const backendPythonDir = app.isPackaged 
      ? path.join(backendDir, 'backend_python')
      : '';

    const args = app.isPackaged
      ? ['-m', 'tyqa.cli', 'deploy', '--port', BACKEND_PORT.toString()]
      : ['run', 'tyqa', 'deploy', '--port', BACKEND_PORT.toString()];

    console.log('Starting Python backend:', pythonExe, args.join(' '));

    const env = { ...process.env };
    if (app.isPackaged) {
      env.PATH = `${backendScriptsDir};${backendPythonDir};${env.PATH || ''}`;
    }

    let backendErrorLog = '';
    let isDead = false;
    let portReady = false;

    backendProcess = spawn(pythonExe, args, {
      cwd: backendDir,
      env: env,
      stdio: 'pipe',
      windowsHide: true // Hide terminal window on Windows
    });

    backendProcess.stdout.on('data', (data) => {
      console.log(`Backend stdout: ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
      const str = data.toString();
      backendErrorLog += str;
      console.error(`Backend stderr: ${str}`);
    });

    backendProcess.on('exit', (code) => {
      isDead = true;
      if (!portReady) {
        const logPath = path.join(app.getPath('userData'), 'backend_crash.log');
        fs.writeFileSync(logPath, backendErrorLog || 'No error log. Exit code: ' + code);
      }
    });

    backendProcess.on('error', (err) => {
      isDead = true;
      console.error('Failed to start backend:', err);
      dialog.showErrorBox('启动失败', '无法启动 Python 后端进程。\\n' + err.message);
      app.quit();
      reject(err);
    });

    // Wait for the backend port to be ready
    const checkPort = () => {
      if (isDead) return reject(new Error(`Backend process died before port was ready.\nBackend Log:\n${backendErrorLog.substring(0, 800)}`));
      const socket = new net.Socket();
      socket.setTimeout(1000);
      socket.on('connect', () => {
        portReady = true;
        socket.destroy();
        resolve();
      });
      socket.on('timeout', () => {
        socket.destroy();
        setTimeout(checkPort, 500);
      });
      socket.on('error', () => {
        socket.destroy();
        setTimeout(checkPort, 500);
      });
      socket.connect(BACKEND_PORT, '127.0.0.1');
    };

    setTimeout(checkPort, 1000);
  });
}

function startServer() {
  return new Promise((resolve, reject) => {
    const frontendDir = app.isPackaged 
      ? path.join(process.resourcesPath, 'frontend') 
      : path.join(__dirname, 'frontend', 'dist');
      
    const serverPath = path.join(frontendDir, 'server.js');
    console.log('Starting frontend server at:', serverPath);
    
    serverProcess = fork(serverPath, [], {
      cwd: frontendDir,
      env: {
        ...process.env,
        PORT: PORT.toString(),
        NODE_ENV: 'production'
      },
      stdio: 'inherit'
    });

    let isServerDead = false;
    let serverPortReady = false;

    serverProcess.on('exit', (code) => {
      isServerDead = true;
    });

    serverProcess.on('error', (err) => {
      isServerDead = true;
      console.error('Failed to start server:', err);
      reject(err);
    });

    const checkPort = () => {
      if (isServerDead) return reject(new Error("Server process died before port was ready"));
      const socket = new net.Socket();
      socket.setTimeout(1000);
      socket.on('connect', () => {
        serverPortReady = true;
        socket.destroy();
        resolve();
      });
      socket.on('timeout', () => {
        socket.destroy();
        setTimeout(checkPort, 500);
      });
      socket.on('error', () => {
        socket.destroy();
        setTimeout(checkPort, 500);
      });
      socket.connect(PORT, '127.0.0.1');
    };

    setTimeout(checkPort, 500);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    title: 'Tianyan Quantum AI',
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  mainWindow.loadURL(`http://127.0.0.1:${PORT}`);

  mainWindow.webContents.on('did-finish-load', () => {
    if (splashWindow && !splashWindow.isDestroyed()) {
      splashWindow.close();
      splashWindow = null;
    }
    mainWindow.webContents.executeJavaScript(`
      const currentConfig = JSON.parse(localStorage.getItem('evoscientist-config') || '{}');
      if (currentConfig.deploymentUrl !== 'http://127.0.0.1:6174') {
        localStorage.setItem('evoscientist-config', JSON.stringify({
          deploymentUrl: 'http://127.0.0.1:6174',
          assistantId: '66c03bd1-fe8c-532f-b0cd-f852a7b58d9a'
        }));
        window.location.reload();
      }
    `);
  });

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  try {
    createSplashWindow();
    console.log('Starting local Python backend...');
    await startBackend();
    console.log('Starting local Node frontend...');
    await startServer();
    console.log('All services ready!');
    createWindow();
  } catch (error) {
    console.error('Could not start services:', error);
    dialog.showErrorBox('启动失败', `无法启动服务:\n${error.message}\n\n请截图此窗口发给开发者。`);
    app.quit();
  }

  app.on('activate', function () {
    if (mainWindow === null) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});

app.on('quit', () => {
  if (serverProcess) {
    serverProcess.kill();
  }
  if (backendProcess) {
    backendProcess.kill();
  }
});
