/**
 * Main JavaScript file for Aura Document Analyzer web interface.
 * Handles file uploads, API interactions, and UI updates with enhanced UX.
 */

// Global configuration
const CONFIG = {
    API_BASE_URL: '',
    MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
    SUPPORTED_TYPES: ['pdf', 'docx', 'json', 'txt'],
    POLLING_INTERVAL: 2000, // 2 seconds
    NOTIFICATION_TIMEOUT: 5000 // 5 seconds
};

// Utility functions
const Utils = {
    /**
     * Format file size in human readable format
     */
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Get file extension from filename
     */
    getFileExtension: function(filename) {
        return filename.split('.').pop().toLowerCase();
    },

    /**
     * Check if file type is supported
     */
    isFileTypeSupported: function(filename) {
        const ext = this.getFileExtension(filename);
        return CONFIG.SUPPORTED_TYPES.includes(ext);
    },

    /**
     * Show notification message with auto-dismiss
     */
    showNotification: function(message, type = 'info', duration = CONFIG.NOTIFICATION_TIMEOUT) {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(n => n.remove());
        
        const notification = document.createElement('div');
        notification.className = `notification alert alert-${type} fixed top-4 right-4 z-50 max-w-sm shadow-lg transform transition-all duration-300 translate-x-full`;
        notification.innerHTML = `
            <div class="flex items-center justify-between">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-lg leading-none">&times;</button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 100);
        
        // Auto remove
        setTimeout(() => {
            if (notification.parentNode) {
                notification.classList.add('translate-x-full');
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }
        }, duration);
    },

    /**
     * Create loading spinner element
     */
    createSpinner: function() {
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        return spinner;
    },

    /**
     * Debounce function to limit API calls
     */
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Validate text content
     */
    validateText: function(text) {
        if (!text || typeof text !== 'string') {
            return { valid: false, error: 'Text content is required' };
        }
        
        if (text.trim().length < 10) {
            return { valid: false, error: 'Text must be at least 10 characters long' };
        }
        
        if (text.length > 1000000) { // 1MB text limit
            return { valid: false, error: 'Text content is too large (max 1MB)' };
        }
        
        return { valid: true };
    }
};

// Enhanced API interaction functions
const API = {
    /**
     * Upload file to the server with progress tracking
     */
    uploadFile: async function(file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const xhr = new XMLHttpRequest();
            
            return new Promise((resolve, reject) => {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        try {
                            const response = JSON.parse(xhr.responseText);
                            resolve(response);
                        } catch (e) {
                            reject(new Error('Invalid JSON response'));
                        }
                    } else {
                        reject(new Error(`Upload failed: ${xhr.statusText}`));
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Network error during upload'));
                });

                xhr.addEventListener('timeout', () => {
                    reject(new Error('Upload timeout'));
                });

                xhr.open('POST', '/api/v1/documents/upload');
                xhr.timeout = 300000; // 5 minutes timeout
                xhr.send(formData);
            });
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    },

    /**
     * Analyze text with AI with retry logic
     */
    analyzeText: async function(text, options = {}, retries = 3) {
        const validation = Utils.validateText(text);
        if (!validation.valid) {
            throw new Error(validation.error);
        }

        const payload = {
            text: text,
            include_classification: options.classification !== false,
            include_ner: options.ner !== false,
            include_embeddings: options.embeddings === true
        };

        for (let attempt = 1; attempt <= retries; attempt++) {
            try {
                const response = await fetch('/api/v1/ai/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload),
                    timeout: 120000 // 2 minutes timeout
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Analysis failed: ${response.status} ${errorText}`);
                }

                return await response.json();
            } catch (error) {
                console.error(`Analysis attempt ${attempt} failed:`, error);
                
                if (attempt === retries) {
                    throw error;
                }
                
                // Wait before retry (exponential backoff)
                await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
            }
        }
    },

    /**
     * Get system health status
     */
    getHealth: async function() {
        try {
            const response = await fetch('/api/v1/health', { timeout: 10000 });
            if (!response.ok) {
                throw new Error(`Health check failed: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Health check error:', error);
            throw error;
        }
    },

    /**
     * Get system statistics
     */
    getStats: async function() {
        try {
            const response = await fetch('/api/v1/stats', { timeout: 10000 });
            if (!response.ok) {
                throw new Error(`Stats request failed: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Stats error:', error);
            throw error;
        }
    }
};

// Enhanced file upload handler
class FileUploadHandler {
    constructor(uploadAreaId, fileInputId) {
        this.uploadArea = document.getElementById(uploadAreaId);
        this.fileInput = document.getElementById(fileInputId);
        this.isUploading = false;
        this.setupEventListeners();
    }

    setupEventListeners() {
        if (!this.uploadArea || !this.fileInput) return;

        // Click to upload
        this.uploadArea.addEventListener('click', (e) => {
            if (!this.isUploading) {
                this.fileInput.click();
            }
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });

        // Enhanced drag and drop with visual feedback
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            // Only remove dragover if we're leaving the upload area entirely
            if (!this.uploadArea.contains(e.relatedTarget)) {
                this.uploadArea.classList.remove('dragover');
            }
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.uploadArea.classList.remove('dragover');
            
            if (!this.isUploading) {
                this.handleFiles(e.dataTransfer.files);
            }
        });

        // Prevent default drag behaviors on document
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
    }

    async handleFiles(files) {
        if (files.length === 0) return;

        const file = files[0];
        
        // Validate file
        const validation = this.validateFile(file);
        if (!validation.valid) {
            Utils.showNotification(validation.error, 'error');
            return;
        }

        try {
            this.isUploading = true;
            this.updateUploadUI(true);
            
            // Upload file with progress tracking
            const uploadResult = await API.uploadFile(file, (progress) => {
                this.updateProgress(progress);
            });
            
            Utils.showNotification('File uploaded successfully', 'success');
            
            // Trigger analysis if on analyze page
            if (window.location.pathname === '/analyze' && window.DocumentAnalyzer) {
                await window.DocumentAnalyzer.processUploadResult(uploadResult);
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            Utils.showNotification(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.isUploading = false;
            this.updateUploadUI(false);
        }
    }

    validateFile(file) {
        // Check file size
        if (file.size > CONFIG.MAX_FILE_SIZE) {
            return {
                valid: false,
                error: `File too large. Maximum size is ${Utils.formatFileSize(CONFIG.MAX_FILE_SIZE)}`
            };
        }

        // Check file type
        if (!Utils.isFileTypeSupported(file.name)) {
            return {
                valid: false,
                error: `Unsupported file type. Supported types: ${CONFIG.SUPPORTED_TYPES.join(', ').toUpperCase()}`
            };
        }

        // Check for empty files
        if (file.size === 0) {
            return {
                valid: false,
                error: 'File is empty'
            };
        }

        return { valid: true };
    }

    updateUploadUI(isUploading) {
        const progressContainer = document.getElementById('upload-progress');
        const uploadArea = this.uploadArea;
        
        if (progressContainer) {
            if (isUploading) {
                progressContainer.classList.remove('hidden');
            } else {
                setTimeout(() => {
                    progressContainer.classList.add('hidden');
                }, 2000);
            }
        }
        
        if (uploadArea) {
            if (isUploading) {
                uploadArea.classList.add('opacity-50', 'pointer-events-none');
            } else {
                uploadArea.classList.remove('opacity-50', 'pointer-events-none');
            }
        }
    }

    updateProgress(percentage) {
        const progressBar = document.getElementById('progress-bar');
        const uploadStatus = document.getElementById('upload-status');
        
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }
        
        if (uploadStatus) {
            if (percentage < 100) {
                uploadStatus.textContent = `Uploading... ${Math.round(percentage)}%`;
            } else {
                uploadStatus.textContent = 'Processing...';
            }
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize file upload handler if elements exist
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    
    if (uploadArea && fileInput) {
        new FileUploadHandler('upload-area', 'file-input');
    }

    // Add smooth scrolling to anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add loading states to buttons
    document.querySelectorAll('button[type="submit"], .btn-primary').forEach(button => {
        button.addEventListener('click', function() {
            if (!this.disabled) {
                const originalText = this.textContent;
                this.disabled = true;
                this.innerHTML = `${Utils.createSpinner().outerHTML} Processing...`;
                
                // Re-enable after 10 seconds as fallback
                setTimeout(() => {
                    this.disabled = false;
                    this.textContent = originalText;
                }, 10000);
            }
        });
    });

    // Add form validation
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function(e) {
            const requiredFields = this.querySelectorAll('[required]');
            let isValid = true;
            
            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    isValid = false;
                    field.classList.add('border-red-500');
                    Utils.showNotification(`${field.name || 'Field'} is required`, 'error');
                } else {
                    field.classList.remove('border-red-500');
                }
            });
            
            if (!isValid) {
                e.preventDefault();
            }
        });
    });
});

// Export for use in other scripts
window.Utils = Utils;
window.API = API;
window.FileUploadHandler = FileUploadHandler;
