const { execSync } = require('child_process');
const path = require('path');

try {
  require('sqlite3');
} catch (err) {
  if (err?.code !== 'ERR_DLOPEN_FAILED') {
    throw err;
  }

  console.log(
    'sqlite3 prebuilt binary is incompatible with this system; compiling from source...'
  );
  execSync('npm rebuild sqlite3 --build-from-source', {
    cwd: path.join(__dirname, '..'),
    stdio: 'inherit',
    env: { ...process.env, npm_config_build_from_source: 'true' },
  });
}
