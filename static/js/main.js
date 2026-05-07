document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('audio-file');
    const loadingSection = document.getElementById('loading');
    const resultsSection = document.getElementById('results');
    const genreName = document.getElementById('genre-name');
    const confidenceScore = document.getElementById('confidence-score');
    const spectrogramImg = document.getElementById('spectrogram-img');
    const resetBtn = document.getElementById('reset-btn');

    // Drag and drop event listeners
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', handleDrop, false);
    
    // Click to upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            handleFiles(this.files[0]);
        }
    });

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        if (files.length > 0) {
            handleFiles(files[0]);
        }
    }

    function handleFiles(file) {
        // Validate file type
        const validTypes = ['audio/wav', 'audio/mpeg', 'audio/basic']; // basic is for .au
        const extension = file.name.split('.').pop().toLowerCase();
        
        if (!validTypes.includes(file.type) && !['wav', 'mp3', 'au'].includes(extension)) {
            alert('Please upload a valid audio file (.wav, .mp3, .au)');
            return;
        }

        uploadFile(file);
    }

    function uploadFile(file) {
        const url = '/predict';
        const formData = new FormData();
        formData.append('file', file);

        // Show loading state
        dropZone.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        fetch(url, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingSection.classList.add('hidden');
            
            if (data.error) {
                alert(data.error);
                dropZone.classList.remove('hidden');
                return;
            }

            // Update UI with results
            genreName.textContent = data.genre;
            confidenceScore.textContent = data.confidence;
            
            // Update audio playback
            const songTitle = document.getElementById('song-title');
            const audioPlayer = document.getElementById('audio-player');
            const audioSource = document.getElementById('audio-source');
            
            if (songTitle && data.song_name) {
                songTitle.textContent = data.song_name;
            }
            if (audioPlayer && audioSource && data.audio_url) {
                audioSource.src = data.audio_url;
                audioPlayer.load(); // reload the audio element
            }
            
            // Add a cache buster parameter to force image reload
            spectrogramImg.src = data.spectrogram_url + '?t=' + new Date().getTime();
            
            // Determine confidence color
            const confValue = parseFloat(data.confidence);
            if (confValue >= 80) {
                confidenceScore.style.color = '#4caf50'; // Green
            } else if (confValue >= 50) {
                confidenceScore.style.color = '#ff9800'; // Orange
            } else {
                confidenceScore.style.color = '#f44336'; // Red
            }

            // Animate result entry
            resultsSection.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during prediction. Please check the console.');
            loadingSection.classList.add('hidden');
            dropZone.classList.remove('hidden');
        });
    }

    resetBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        fileInput.value = '';
        dropZone.classList.remove('hidden');
    });
});
