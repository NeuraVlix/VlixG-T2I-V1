document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const promptInput = document.getElementById('prompt');
    const generatedImage = document.getElementById('generated-image');
    const loadingIndicator = document.getElementById('loading-indicator');

    generateBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        
        if (!prompt) {
            alert('Please enter a description');
            return;
        }
        
        // Show loading state
        generatedImage.style.display = 'none';
        loadingIndicator.style.display = 'block';
        
        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: prompt })
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate image');
            }
            
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);
            
            generatedImage.src = imageUrl;
            generatedImage.style.display = 'block';
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error generating image: ' + error.message);
        } finally {
            loadingIndicator.style.display = 'none';
        }
    });
});
