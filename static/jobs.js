class JobRecommendations {
    constructor() {
        this.jobs = [];
        this.filteredJobs = [];
    }

    async fetchJobs() {
        try {
            const response = await fetch('/get_job_recommendations');
            const data = await response.json();
            
            if (data.error) {
                console.error(data.error);
                return;
            }
            
            this.jobs = data.recommendations.map(job => ({
                title: job.title,
                company: 'Various Companies',
                location: 'Multiple Locations',
                experience: 'All Levels',
                salary: job.salary,
                description: job.description,
                job_type: job.job_type
            }));
            
            this.filteredJobs = [...this.jobs];
            this.renderJobs();
        } catch (error) {
            console.error('Error fetching job recommendations:', error);
        }
    }

    filterJobs() {
        const location = document.getElementById('locationFilter').value;
        const experience = document.getElementById('experienceFilter').value;

        this.filteredJobs = this.jobs.filter(job => {
            return (!location || job.location === location) &&
                   (!experience || job.experience === experience);
        });

        this.renderJobs();
    }

    renderJobs() {
        const jobList = document.getElementById('jobList');
        if (this.filteredJobs.length === 0) {
            jobList.innerHTML = '<div class="no-jobs">No job recommendations found. Please upload your resume first.</div>';
            return;
        }
        
        jobList.innerHTML = this.filteredJobs.map(job => `
            <div class="job-card">
                <h3>${job.title}</h3>
                <div class="company-info">
                    <p>${job.company}</p>
                    <p>Location: ${job.location}</p>
                    <p>Job Type: ${job.job_type}</p>
                </div>
                <p class="salary">Salary Range: ${job.salary}</p>
                <p class="description">${job.description}</p>
                <button class="apply-btn" onclick="jobSystem.applyForJob('${job.title}')">
                    Apply Now
                </button>
            </div>
        `).join('');
    }

    applyForJob(jobTitle) {
        alert(`Applied for ${jobTitle}`);
    }
}

const jobSystem = new JobRecommendations();
document.addEventListener('DOMContentLoaded', () => {
    jobSystem.fetchJobs();
    document.getElementById('locationFilter').addEventListener('change', () => jobSystem.filterJobs());
    document.getElementById('experienceFilter').addEventListener('change', () => jobSystem.filterJobs());
}); 