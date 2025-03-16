// Initialize charts
let salaryChart = null;
let skillsChart = null;
let industryChart = null;
let locationChart = null;

function setLoading(chartId, isLoading) {
    const container = document.getElementById(chartId).closest('.chart-container');
    const overlay = container.querySelector('.loading-overlay');
    if (isLoading) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}

function setButtonLoading(isLoading) {
    const button = document.querySelector('.analyze-btn');
    if (isLoading) {
        button.classList.add('loading');
        button.disabled = true;
    } else {
        button.classList.remove('loading');
        button.disabled = false;
    }
}

function setAllChartsLoading(isLoading) {
    setLoading('salaryChart', isLoading);
    setLoading('skillsChart', isLoading);
    setLoading('industryChart', isLoading);
    setLoading('locationChart', isLoading);
    setButtonLoading(isLoading);
}

let currentRequest = null;

async function fetchWithRetry(url, options, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            
            // Wait before retrying (exponential backoff)
            const waitTime = Math.min(1000 * Math.pow(2, i), 5000);
            await new Promise(resolve => setTimeout(resolve, waitTime));
            
            // Update UI to show retry attempt
            const button = document.querySelector('.analyze-btn');
            const spinnerText = button.querySelector('.button-text');
            spinnerText.textContent = `Retrying... (${i + 1}/${maxRetries})`;
        }
    }
}

async function analyzeAllMetrics() {
    const jobTitle = document.getElementById('jobTitle').value;
    if (!jobTitle) {
        alert('Please enter a job title');
        return;
    }

    // Cancel previous request if it exists
    if (currentRequest) {
        currentRequest.abort();
    }

    setAllChartsLoading(true);
    const formData = new FormData();
    formData.append('jobTitle', jobTitle);

    // Create AbortController for the new request
    const controller = new AbortController();
    currentRequest = controller;

    try {
        const data = await fetchWithRetry('/analyze_job_market', {
            method: 'POST',
            body: formData,
            signal: controller.signal,
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        });

        if (data.error) {
            throw new Error(data.error);
        }

        // Update charts as data becomes available
        if (data.salary && !data.salary.error) {
            createSalaryChart({
                ...data.salary,
                job_title: data.job_title
            });
            setLoading('salaryChart', false);
        } else if (data.salary?.error) {
            console.error('Salary analysis error:', data.salary.error);
            setLoading('salaryChart', false);
            showChartError('salaryChart', 'Unable to load salary data');
        }

        if (data.competition && !data.competition.error) {
            createJobCompetitionChart({
                ...data.competition,
                job_title: data.job_title
            });
            setLoading('skillsChart', false);
        } else if (data.competition?.error) {
            console.error('Competition analysis error:', data.competition.error);
            setLoading('skillsChart', false);
            showChartError('skillsChart', 'Unable to load competition data');
        }

        if (data.industry && !data.industry.error) {
            createCompaniesChart({
                companies: data.industry.companies,
                job_title: data.job_title,
                total_jobs: data.industry.total_jobs
            });
            createLocationsChart({
                locations: data.industry.locations,
                job_title: data.job_title,
                total_jobs: data.industry.total_jobs
            });
            setLoading('industryChart', false);
            setLoading('locationChart', false);
        } else if (data.industry?.error) {
            console.error('Industry analysis error:', data.industry.error);
            setLoading('industryChart', false);
            setLoading('locationChart', false);
            showChartError('industryChart', 'Unable to load industry data');
            showChartError('locationChart', 'Unable to load location data');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Request cancelled');
        } else {
            console.error('Error in analysis:', error);
            const errorMessage = error.message === 'Failed to fetch' 
                ? 'Unable to connect to the server. Please check your internet connection and try again.'
                : `An error occurred: ${error.message}`;
            
            alert(errorMessage);
            setAllChartsLoading(false);
            showAllChartsError('Unable to load data. Please try again.');
        }
    } finally {
        if (currentRequest === controller) {
            currentRequest = null;
            setButtonLoading(false);
        }
    }
}

function createCompaniesChart(data) {
    if (industryChart) {
        industryChart.destroy();
    }

    const ctx = document.getElementById('industryChart').getContext('2d');
    industryChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.companies.names,
            datasets: [{
                label: 'Number of Job Postings',
                data: data.companies.counts,
                backgroundColor: Array(data.companies.names.length).fill('rgba(54, 162, 235, 0.7)'),
                borderColor: Array(data.companies.names.length).fill('rgba(54, 162, 235, 1)'),
                borderWidth: 1,
                barPercentage: 0.8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: [
                        `Top Hiring Companies for ${data.job_title}`,
                        `Based on ${data.total_jobs} recent job postings`
                    ],
                    font: {
                        size: 14,
                        weight: 'bold'
                    },
                    padding: 20
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.x} job postings`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Number of Job Postings',
                        font: {
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        precision: 0
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Company',
                        font: {
                            weight: 'bold'
                        }
                    }
                }
            }
        }
    });
}

function createLocationsChart(data) {
    if (locationChart) {
        locationChart.destroy();
    }

    const ctx = document.getElementById('locationChart').getContext('2d');
    locationChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.locations.names,
            datasets: [{
                label: 'Number of Job Postings',
                data: data.locations.counts,
                backgroundColor: Array(data.locations.names.length).fill('rgba(75, 192, 192, 0.7)'),
                borderColor: Array(data.locations.names.length).fill('rgba(75, 192, 192, 1)'),
                borderWidth: 1,
                barPercentage: 0.8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: [
                        `Top Job Locations for ${data.job_title}`,
                        `Based on ${data.total_jobs} recent job postings`
                    ],
                    font: {
                        size: 14,
                        weight: 'bold'
                    },
                    padding: 20
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.parsed.x} job postings`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Number of Job Postings',
                        font: {
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        precision: 0
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Location',
                        font: {
                            weight: 'bold'
                        }
                    }
                }
            }
        }
    });
}

function createSalaryChart(data) {
    if (salaryChart) {
        salaryChart.destroy();
    }

    // Format dates and prepare datasets
    const historicalData = data.historical_data.map(item => ({
        x: new Date(item.ds).getTime(),
        y: Math.round(item.y)
    }));

    const forecastData = data.forecast_data.map(item => ({
        x: new Date(item.ds).getTime(),
        y: Math.round(item.yhat),
        yLower: Math.round(item.yhat_lower),
        yUpper: Math.round(item.yhat_upper)
    }));

    // Split forecast data into historical and future periods
    const currentDate = new Date().getTime();
    const historicalForecast = forecastData.filter(d => d.x <= currentDate);
    const futureForecast = forecastData.filter(d => d.x > currentDate);

    const ctx = document.getElementById('salaryChart').getContext('2d');
    salaryChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Historical Salary Data',
                    data: historicalData,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: false
                },
                {
                    label: 'Model Fit',
                    data: historicalForecast,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Salary Forecast',
                    data: futureForecast,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Salary Trends for ${data.job_title}`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', {
                                    style: 'currency',
                                    currency: 'USD',
                                    minimumFractionDigits: 0,
                                    maximumFractionDigits: 0
                                }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'month',
                        tooltipFormat: 'MMM yyyy'
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Average Salary (USD)'
                    },
                    ticks: {
                        callback: function(value) {
                            return new Intl.NumberFormat('en-US', {
                                style: 'currency',
                                currency: 'USD',
                                minimumFractionDigits: 0,
                                maximumFractionDigits: 0
                            }).format(value);
                        }
                    }
                }
            }
        }
    });
}

function createJobCompetitionChart(data) {
    if (skillsChart) {
        skillsChart.destroy();
    }

    const ctx = document.getElementById('skillsChart').getContext('2d');
    
    // Calculate total for percentages
    const total = data.work_type.values.reduce((a, b) => a + b, 0);
    
    // Create a doughnut chart with two datasets side by side
    skillsChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: data.work_type.values,
                backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)'
                ],
                label: 'Work Type'
            }],
            labels: data.work_type.labels
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: [
                        `Job Competition Analysis for ${data.job_title}`,
                        `Based on ${data.total_jobs} job postings`
                    ],
                    font: {
                        size: 14,
                        weight: 'bold'
                    },
                    padding: 20
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        generateLabels: function(chart) {
                            const data = chart.data;
                            if (data.labels.length && data.datasets.length) {
                                return data.labels.map((label, i) => {
                                    const value = data.datasets[0].data[i];
                                    const percentage = ((value / total) * 100).toFixed(1);
                                    return {
                                        text: `${label}: ${value} (${percentage}%)`,
                                        fillStyle: data.datasets[0].backgroundColor[i],
                                        hidden: isNaN(data.datasets[0].data[i]),
                                        index: i
                                    };
                                });
                            }
                            return [];
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed;
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${context.label}: ${value} jobs (${percentage}%)`;
                        }
                    }
                },
                datalabels: {
                    display: true,
                    color: '#fff',
                    font: {
                        weight: 'bold',
                        size: 12
                    },
                    formatter: function(value) {
                        const percentage = ((value / total) * 100).toFixed(1);
                        return `${percentage}%`;
                    }
                }
            },
            layout: {
                padding: {
                    top: 20,
                    bottom: 20
                }
            }
        },
        plugins: [{
            afterDraw: function(chart) {
                const ctx = chart.ctx;
                ctx.save();
                const centerX = chart.chartArea.left + (chart.chartArea.right - chart.chartArea.left) / 2;
                const centerY = chart.chartArea.top + (chart.chartArea.bottom - chart.chartArea.top) / 2;
                
                ctx.fillStyle = '#666';
                ctx.font = '14px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(`Total: ${total} jobs`, centerX, centerY);
                ctx.restore();
            }
        }]
    });
}

function showChartError(chartId, message) {
    const container = document.getElementById(chartId).closest('.chart-container');
    const overlay = container.querySelector('.loading-overlay');
    const spinner = overlay.querySelector('.fa-spinner');
    const text = overlay.querySelector('span');
    
    spinner.style.display = 'none';
    text.textContent = message;
    text.style.color = '#dc3545';
    overlay.classList.add('active');
}

function showAllChartsError(message) {
    ['salaryChart', 'skillsChart', 'industryChart', 'locationChart'].forEach(chartId => {
        showChartError(chartId, message);
    });
}

// Debounce function to prevent too many API calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Update the event listeners
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('jobAnalysisForm');
    const jobTitleInput = document.getElementById('jobTitle');

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            analyzeAllMetrics();
        });
    }

    if (jobTitleInput) {
        const debouncedAnalysis = debounce(analyzeAllMetrics, 500);
        jobTitleInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                analyzeAllMetrics();
            }
        });
    }
});
