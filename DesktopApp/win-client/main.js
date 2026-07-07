const { app, BrowserWindow } = require('electron');
const { fork, spawn } = require('child_process');
const path = require('path');
const net = require('net');

let mainWindow;
let serverProcess;
let backendProcess;
const PORT = 4716;
const BACKEND_PORT = 6174;

function startBackend() {
  return new Promise((resolve, reject) => {
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
      console.error(`Backend stderr: ${data}`);
    });

    backendProcess.on('error', (err) => {
      console.error('Failed to start backend:', err);
      reject(err);
    });

    // Wait for the backend port to be ready
    const checkPort = () => {
      const socket = new net.Socket();
      socket.setTimeout(1000);
      socket.on('connect', () => {
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

    serverProcess.on('error', (err) => {
      console.error('Failed to start server:', err);
      reject(err);
    });

    const checkPort = () => {
      const socket = new net.Socket();
      socket.setTimeout(1000);
      socket.on('connect', () => {
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
    console.log('Starting local Python backend...');
    await startBackend();
    console.log('Starting local Node frontend...');
    await startServer();
    console.log('All services ready!');
    createWindow();
  } catch (error) {
    console.error('Could not start services:', error);
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
