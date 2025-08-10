document.querySelector('form').addEventListener('submit', (e) => {
    const spinner = document.querySelector('.loading-spinner');
    spinner.style.display = 'inline-block';
  });

  document.getElementById('stopBtn').addEventListener('click', () => {
    fetch('/stop', { method: 'POST' });
    document.getElementById('liveFeed').src = '';
  });

  document.getElementById('scanBtn').addEventListener('click', () => {
    fetch('/restart_feed', { method: 'POST' }).then(() => {
      document.getElementById('liveFeed').src = '/video_feed';
    });
  });

  setInterval(() => {
    fetch('/check_attendance')
      .then(res => res.json())
      .then(data => {
        if (data.marked) {
          document.getElementById("attendanceMessage").innerText = "Attendance Marked ✅";
        }
      });
  }, 2000); 
