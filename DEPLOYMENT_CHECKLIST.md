# Deployment Checklist

## Pre-Deployment

- [ ] Code is committed to Git
- [ ] Repository is pushed to GitHub
- [ ] All environment variables documented
- [ ] Model file accessible (in repo or external storage)
- [ ] Backend CORS updated for production domains
- [ ] Frontend API routes configured

## Vercel (Frontend)

- [ ] Account created at vercel.com
- [ ] Project imported from GitHub
- [ ] Root directory set to `frontend`
- [ ] Environment variable `BACKEND_URL` added (after backend deploys)
- [ ] Build successful
- [ ] Frontend URL working

## Railway (Backend)

- [ ] Account created at railway.app
- [ ] Project created from GitHub
- [ ] Root directory set to `backend`
- [ ] Environment variables set:
  - [ ] `PORT=8000`
  - [ ] `PYTHONUNBUFFERED=1`
- [ ] Model file accessible
- [ ] Backend URL obtained
- [ ] Health check endpoint working

## Post-Deployment

- [ ] Frontend `BACKEND_URL` updated in Vercel
- [ ] Frontend redeployed with correct backend URL
- [ ] Test upload functionality
- [ ] Test sample image functionality
- [ ] Check browser console for errors
- [ ] Check Railway logs for backend errors
- [ ] Verify CORS is working
- [ ] Test on mobile device (responsive)

## Optional Enhancements

- [ ] Custom domain configured
- [ ] SSL certificate verified (automatic on Vercel/Railway)
- [ ] Monitoring/analytics set up
- [ ] Error tracking configured
- [ ] CI/CD pipeline set up

## Troubleshooting Commands

```bash
# Check Vercel deployment
vercel logs

# Check Railway logs
# (In Railway dashboard)

# Test backend locally
cd backend && python3 api.py

# Test frontend locally
cd frontend && npm run dev
```
