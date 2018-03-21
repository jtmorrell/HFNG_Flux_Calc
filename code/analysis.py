import matplotlib.pyplot as plt, numpy as np, matplotlib, datetime as dtm
import re,urllib2,os
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.interpolate import interp1d
from numpy.random import rand,randn
import sqlite3
matplotlib.rc('font',**{'size':14,'family':'sans-serif'})
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'
plt.rcParams['legend.fontsize']='12'
_db_connection = sqlite3.connect('../data/nudat.db')
_db = _db_connection.cursor()

class isotope(object):
	def __init__(self,istp):
		self.db_connection = _db_connection
		self.db = _db
		self.tex_g,self.metastable = False,False
		if istp.endswith('m'):
			self.metastable = True
			self.istp = istp[:-1].upper()
		elif istp.endswith('g'):
			self.istp = istp[:-1].upper()
			self.tex_g = True
		else:
			self.istp = istp.upper()
		self.t_half,self.unc_t_half = None,None
	def parse_isotope_dat(self,ln):
		fmt = {}
		for n,el in enumerate(['isotope','E_parent','t_half','energy','intensity','unc_intensity','unc_t_half']):
			fmt[el] = str(ln[n]) if n==0 else float(ln[n])
		fmt['metastable'] = True if fmt['E_parent'] > 0 else False
		return fmt
	def name(self):
		return self.istp+('m' if self.metastable else '')+('g' if self.tex_g else '')
	def TeX(self):
		el = ''.join(re.findall('[A-Z]+',self.istp))
		return r'$^{'+self.istp.split(el)[0]+('m' if self.metastable else '')+('g' if self.tex_g else '')+r'}$'+el.title()
	def gammas(self,Imin=0,Emin=70):
		self.gm = []
		for ln in self.db.execute('SELECT * FROM ensdf WHERE isotope=?',(self.istp,)):
			fmt = self.parse_isotope_dat(ln)
			if fmt['metastable']==self.metastable:
				### NOTE: WILL BE WRONG IF MORE THAN ONE ISOMER!!! ###
				self.t_half,self.unc_t_half = fmt['t_half'],fmt['unc_t_half']
				self.gm.append((fmt['energy'],fmt['intensity'],fmt['unc_intensity']))
		return [g for g in self.gm if g[0]>=Emin and g[1]>=Imin]
	def half_life(self,units='s'):
		if self.t_half is None:
			self.t_half = [self.parse_isotope_dat(ln) for ln in self.db.execute('SELECT * FROM ensdf WHERE isotope=? AND E_parent'+('>0' if self.metastable else '=0'),(self.istp,))][0]['t_half']
		return self.t_half/{'s':1.0,'m':60.0,'h':3600.0,'d':86400.0,'y':3.154e+7}[units]
	def unc_half_life(self,units='s'):
		if self.unc_t_half is None:
			self.unc_t_half = [self.parse_isotope_dat(ln) for ln in self.db.execute('SELECT * FROM ensdf WHERE isotope=? AND E_parent'+('>0' if self.metastable else '=0'),(self.istp,))][0]['unc_t_half']
		return self.unc_t_half/{'s':1.0,'m':60.0,'h':3600.0,'d':86400.0,'y':3.154e+7}[units]
	def decay_const(self,units='s'):
		return np.log(2)/self.half_life(units)
	def unc_decay_const(self,units='s'):
		return np.log(2)*self.unc_half_life(units)/self.half_life(units)**2
	def is_metastable(self):
		return self.metastable

class flux_calc(object):
	def __init__(self,directory=None):
		self.db_connection = _db_connection
		self.db = _db
		self.pallate = {'k':'#2c3e50','b':'#2980b9','r':'#c0392b','y':'#f39c12','p':'#8e44ad','g':'#27ae60','gy':'#7f8c8d','o':'#d35400','w':'#ecf0f1','aq':'#16a085'}
		self.sp = {'E0_D2':102.6,'dx':0.0,'dy':0.0,'dz':9.5,'R_src':7.0,'R_smpl':5.5,'elbow':False}
		self.directory = directory
		self.spec = None
	def get_energy_spectrum(self,Nps=1e5):
		if self.spec is not None:
			return self.spec[0],self.spec[1],self.spec[2]
		E0_D2,dx,dy,dz,R_src,R_smpl,elbow = self.sp['E0_D2'],self.sp['dx'],self.sp['dy'],self.sp['dz'],self.sp['R_src'],self.sp['R_smpl'],self.sp['elbow']
		A100,A200 = [2.46674,0.30083,0.01368,0.0],[2.47685,0.39111,0.04098,0.02957]
		B100,B200 = [0.01741,0.88746,0.22497,0.08183,0.37225],[-0.03149,1.11225,0.38659,0.26676,0.11518]
		coeffA = [A100[n]+(E0_D2-100.0)*(A200[n]-A100[n])/100.0 for n in range(4)]
		coeffB = [B100[n]+(E0_D2-100.0)*(B200[n]-B100[n])/100.0 for n in range(5)]
		eng_func = lambda theta_deg: coeffA[0]+sum([coeffA[n]*np.cos(theta_deg*np.pi/180.0)**n for n in range(1,4)])
		yield_func = lambda theta_deg: 1.0+sum([coeffB[n]*np.cos(theta_deg*np.pi/180.0)**n for n in range(5)])
		#select src position (Gaussian)
		r_src,tht_src = R_src*0.33*randn(int(Nps)),2.0*np.pi*rand(int(Nps))
		xy_src = [(abs(r)*np.cos(tht_src[n]),abs(r)*np.sin(tht_src[n])) for n,r in enumerate(r_src)]
		#select sample position (uniform)
		r_smpl,tht_smpl = R_smpl*np.sqrt(rand(int(Nps))),2.0*np.pi*rand(int(Nps))
		xy_smpl = [(dx+r*np.cos(tht_smpl[n]),dy+r*np.sin(tht_smpl[n])) for n,r in enumerate(r_smpl)]
		#calculate angle/energy
		if elbow:
			angle_deg = [180.0*np.arccos((dx-xy_smpl[n][0])/np.sqrt((dz-s[0])**2+(xy_smpl[n][1]-s[1])**2+(dx-xy_smpl[n][0])**2))/np.pi for n,s in enumerate(xy_src)]
		else:
			angle_deg = [180.0*np.arccos(dz/np.sqrt((xy_smpl[n][0]-s[0])**2+(xy_smpl[n][1]-s[1])**2+dz**2))/np.pi for n,s in enumerate(xy_src)]
		E_n = list(map(eng_func,angle_deg))
		#calculate weight
		dist = [np.sqrt(dz**2+((xy_smpl[n][0]-s[0])**2+(xy_smpl[n][1]-s[1])**2)) for n,s in enumerate(xy_src)]
		wts = [yield_func(angle_deg[n])/d**2 for n,d in enumerate(dist)]
		#sum wts
		hist,bin_edges = np.histogram(E_n,bins='auto')
		mu_E,sig_E,L = np.average(E_n),np.std(E_n),len(bin_edges)
		hist,unc = np.zeros(L-1),np.zeros(L-1)
		for n,e in enumerate(E_n):
			m = max((min((int(L*(0.5+(e-mu_E)/(6.0*sig_E))),L-2)),0))
			while bin_edges[m+1]<e and -1<m<L:
				m += 1
			while bin_edges[m]>=e and -1<m<L:
				m -= 1
			hist[m] += wts[n]
			unc[m] += wts[n]**2
		self.spec = (bin_edges,hist,np.sqrt(unc))
		return bin_edges,hist,np.sqrt(unc)
	def set_sample_params(self,E0_D2=102.6,dx=0.0,dy=0.0,dz=9.5,R_src=7.0,R_smpl=5.5,elbow=False):
		self.sp = {'E0_D2':E0_D2,'dx':dx,'dy':dy,'dz':dz,'R_src':R_src,'R_smpl':R_smpl,'elbow':elbow}
		self.spec = None
	def plot_energy_spectrum(self,saveplot=True,show=False):
		edges,hist,unc = self.get_energy_spectrum()
		hist,unc = hist/sum(hist),unc/sum(hist)
		x,y = [],[]
		for i in range(len(hist)):
			x.append(edges[i])
			x.append(edges[i+1])
			y.append(hist[i])
			y.append(hist[i])
		mu_E = (1.0/sum(hist))*sum([0.5*(edges[n+1]+edges[n])*h for n,h in enumerate(hist)])
		sig_E = np.sqrt((1.0/sum(hist))*sum([h*(0.5*(edges[n]+edges[n+1])-mu_E)**2 for n,h in enumerate(hist)]))
		# print 'E =',round(mu_E,2),'+/-',round(sig_E,3),'[MeV]'
		self.EdE = [mu_E,sig_E]
		f,ax = plt.subplots()
		ax.plot(x,y,color=self.pallate['k'],lw=2.0)
		ax.errorbar([0.5*(e+edges[n+1]) for n,e in enumerate(edges[:-1])],hist,yerr=2.0*unc,color=self.pallate['k'],ls='None',label='95% Confidence Bands')
		ymx = max(hist)
		ax.vlines(mu_E,ymin=0,ymax=ymx,colors=self.pallate['gy'],lw=2.0,label=r'$E_{average}$')
		ax.vlines([mu_E-sig_E,mu_E+sig_E],ymin=0,ymax=ymx,colors=self.pallate['gy'],lw=2.0,linestyles='--',label=r'$\pm 1\sigma_E$')
		ax.set_xlabel('Neutron Energy [MeV]')
		ax.set_ylabel('Flux [a.u.]')
		ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]+0.008)
		ax.legend(loc=0)
		f.tight_layout()
		if saveplot:
			f.savefig('../plots/'+self.directory+'/monitors/'+str(int(self.sp['dx']))+str(int(self.sp['dy']))+str(int(self.sp['dz']))+str(self.sp['elbow'])+'.png')
			f.savefig('../plots/'+self.directory+'/monitors/'+str(int(self.sp['dx']))+str(int(self.sp['dy']))+str(int(self.sp['dz']))+str(self.sp['elbow'])+'.pdf')
		if show:
			plt.show()
		else:
			plt.close()
	def plot_energy_angle(self,show=False,saveplot=True):
		A100,A200 = [2.46674,0.30083,0.01368,0.0],[2.47685,0.39111,0.04098,0.02957]
		coeff = [A100[n]+(self.sp['E0_D2']-100.0)*(A200[n]-A100[n])/100.0 for n in range(4)]
		f,ax = plt.subplots()
		eng_func = lambda theta_deg: coeff[0]+sum([coeff[n]*np.cos(theta_deg*np.pi/180.0)**n for n in range(1,4)])
		ax.plot(np.arange(0,180,0.1),[eng_func(t) for t in np.arange(0,180,0.1)],color=self.pallate['k'],lw=2.0,label=r'$E_0=$'+str(round(self.sp['E0_D2'],1))+' keV')
		ax.set_xlabel(r'$\theta$ [degrees]')
		ax.set_ylabel('Neutron Energy [MeV]')
		ax.legend(loc=0)
		f.tight_layout()
		if saveplot:
			f.savefig('../plots/'+self.directory+'/monitors/energy_angle.png')
			f.savefig('../plots/'+self.directory+'/monitors/energy_angle.pdf')
		if show:
			plt.show()
		else:
			plt.close()
	def plot_intensity_angle(self,show=False,saveplot=True):
		B100,B200 = [0.01741,0.88746,0.22497,0.08183,0.37225],[-0.03149,1.11225,0.38659,0.26676,0.11518]
		coeffB = [B100[n]+(self.sp['E0_D2']-100.0)*(B200[n]-B100[n])/100.0 for n in range(5)]
		yield_func = lambda theta_deg: 1.0+sum([coeffB[n]*np.cos(theta_deg*np.pi/180.0)**n for n in range(5)])
		f,ax = plt.subplots()
		ax.plot(np.arange(0,180,0.1),[yield_func(t) for t in np.arange(0,180,0.1)],color=self.pallate['k'],lw=2.0,label=r'$E_0=$'+str(round(self.sp['E0_D2'],1))+' keV')
		ax.set_xlabel(r'$\theta$ [degrees]')
		ax.set_ylabel(r'R($\theta$)/R(90$^{\circ}$)')
		ax.legend(loc=0)
		f.tight_layout()
		if saveplot:
			f.savefig('../plots/'+self.directory+'/monitors/intensity_angle.png')
			f.savefig('../plots/'+self.directory+'/monitors/intensity_angle.pdf')
		if show:
			plt.show()
		else:
			plt.close()
	def get_smooth_xs(self,reaction):
		x4 = [(float(i[2]),float(i[3])) for i in self.db.execute('SELECT * FROM monitor_xs WHERE product=?',(reaction.split('_')[-1],))]
		return interp1d([i[0] for i in x4],[i[1] for i in x4])
	def get_error_xs(self,reaction):
		x4 = [(float(i[2]),float(i[4])) for i in self.db.execute('SELECT * FROM monitor_xs WHERE product=?',(reaction.split('_')[-1],))]
		return interp1d([i[0] for i in x4],[i[1] for i in x4])
	def get_average_xs(self,reaction='115IN_n_inl_115INm'):
		mu_sig = self.get_smooth_xs(reaction)
		unc_sig = self.get_error_xs(reaction)
		bin_edges,hist,unc_hist = self.get_energy_spectrum()
		avg_xs = sum([hist[n]*0.5*(mu_sig(e)+mu_sig(bin_edges[n+1])) for n,e in enumerate(bin_edges[:-1])])/sum(hist)
		unc_avg_xs = sum([hist[n]*0.5*(unc_sig(e)+unc_sig(bin_edges[n+1])) for n,e in enumerate(bin_edges[:-1])])/sum(hist)
		return avg_xs,unc_avg_xs

class spectrum(object):
	def __init__(self,filename):
		self.db_connection = _db_connection
		self.db = _db
		meta = [[str(i[1]),str(i[3]).split(';')] for i in self.db.execute('SELECT * FROM experiment_files WHERE filename=?',(filename,))]
		if len(meta)==0:
			meta = [[str(i[1]),[str(i[2])]] for i in self.db.execute('SELECT * FROM calibration_files WHERE filename=?',(filename,))]
		self.directory = meta[0][0]
		self.isotope_list = meta[0][1]
		self.filename = filename.split('.')[0]
		f = open('../data/spectra/'+self.directory+'/'+filename,'r').read().split('\n')
		self.start_time = dtm.datetime.strptime(f[7].replace('\r',''),'%m/%d/%Y %H:%M:%S')
		self.live_time = float(f[9].split(' ')[0].strip())
		self.real_time = float(f[9].split(' ')[1].strip())
		self.spec = [int(f[n].strip()) for n in range(12,13+int(f[11].split(' ')[1]))]
		self.peak_fits = None
		self.calibration = [[float(i[1].split(',')[0]),float(i[1].split(',')[1])] for i in self.db.execute('SELECT * FROM calibration WHERE directory=?',(self.directory,))][0]
		self.resolution = [float(i[3]) for i in self.db.execute('SELECT * FROM calibration WHERE directory=?',(self.directory,))][0]
		self.efficiency = [[float(m) for m in i[2].split(',')] for i in self.db.execute('SELECT * FROM calibration WHERE directory=?',(self.directory,))][0]
		self.pk_args = [[float(i[4]),float(i[5])] for i in self.db.execute('SELECT * FROM calibration WHERE directory=?',(self.directory,))][0]
		self.SNP = self.SNIP()
		self.pallate = {'k':'#2c3e50','b':'#2980b9','r':'#c0392b','y':'#f39c12','p':'#8e44ad','g':'#27ae60','gy':'#7f8c8d','o':'#d35400','w':'#ecf0f1','aq':'#16a085'}
	def get_energy(self):
		a,b = self.calibration[0],self.calibration[1]
		return [a*i+b for i in range(len(self.spec))]
	def get_plot_spec(self,logscale=True,mn=0,mx=None):
		mx = len(self.spec) if mx is None else mx
		lim_y = 1 if logscale else 0
		bn = [max((self.spec[mn],lim_y))]
		for i in range(mn,mx-1):
			bn.append(max((self.spec[i],lim_y)))
			bn.append(max((self.spec[i+1],lim_y)))
		return bn
	def get_plot_energy(self,mn=0,mx=None):
		mx = len(self.spec) if mx is None else mx
		a,b = self.calibration
		eng = [a*(i-0.5)+b for i in range(mn,mx)]
		bn = []
		for i in range(1,len(eng)):
			bn.append(eng[i-1])
			bn.append(eng[i])
		bn.append(eng[-1])
		return bn
	def get_efficiency(self,energy):
		a,b,c = self.efficiency[0],self.efficiency[1],self.efficiency[2]
		return np.exp(a*np.log(energy)**2+b*np.log(energy)+c)
	def set_calibration(self,calibration):
		self.calibration = calibration
	def set_efficiency(self,efficiency):
		self.efficiency = efficiency
	def exp_smooth(self,ls,alpha=0.3):
		R,RR,b = [ls[0]],[ls[-1]],1.0-alpha
		for i,ii in zip(ls[1:],reversed(ls[:-1])):
			R.append(alpha*i+b*R[-1])
			RR.append(alpha*ii+b*RR[-1])
		return [0.5*(R[n]+r) for n,r in enumerate(reversed(RR))]
	def SNIP(self,N=9,alpha=0.2):
		dead=0
		while self.spec[dead]==0:
			dead+=1
		vi = [np.log(np.log(np.sqrt(i+1.0)+1.0)+1.0) for i in self.spec]
		a,L,off = self.resolution,len(vi),int(self.resolution*N*len(vi)**0.5)
		for M in range(1,N):
			vi = vi[:dead+off]+[min((v,0.5*(vi[n-max((M*int(a*n**0.5),M/2))]+vi[n+max((M*int(a*n**0.5),M/2))]))) for v,n in zip(vi[dead+off:L-off],range(dead+off,L-off))]+vi[-1*off:]
		return [s+1.5*np.sqrt(s+1) for n,s in enumerate(self.exp_smooth([int((np.exp(np.exp(i)-1.0)-1.0)**2)-1 for i in vi],alpha=alpha))]
	def find_pks(self,sensitivity=0.5,ls=None):
		N,sg,alpha,Emin = int(10-4*sensitivity),4.5-4.0*sensitivity,0.1+0.5*sensitivity,50
		SNP = self.SNIP(N=N)
		ls = self.spec if ls is None else ls
		clip = [int(i-SNP[n]) if (i-SNP[n])>sg*np.sqrt(SNP[n]) else 0 for n,i in enumerate(self.exp_smooth(ls,alpha=alpha))]
		pks = [n+1 for n,i in enumerate(clip[1:-1]) if all([i>clip[n],i>clip[n+2],clip[n]>0,clip[n+2]>0])]
		return [p for n,p in enumerate(pks[1:]) if pks[n]+3*self.resolution*pks[n]**0.5<p and p*self.calibration[0]+self.calibration[1]>Emin]
	def simple_regression(self,x,y):
		xb,yb = np.average(x),np.average(y)
		m = sum([(i-xb)*(y[n]-yb) for n,i in enumerate(x)])/sum([(i-xb)**2 for i in x])
		return m,yb-m*xb
	def peak(self,x,A,mu,sig):
		r2,R,alpha = 1.41421356237,self.pk_args[0],self.pk_args[1]
		return [A*np.exp(-0.5*((i-mu)/sig)**2)+R*A*np.exp((i-mu)/(alpha*sig))*erfc((i-mu)/(r2*sig)+1.0/(r2*alpha)) for i in x]
	def Npeak(self,x,*args):
		N = (len(args)-2)/3
		I = [self.peak(x,args[2+3*n],args[2+3*n+1],args[2+3*n+2]) for n in range(N)]
		return [args[0]*i+args[1]+sum([I[m][n] for m in range(N)]) for n,i in enumerate(x)]
	def chi2(self,fn,x,b,y):
		if float(len(y)-len(b))<1:
			return float('Inf')
		return sum([(y[n]-i)**2/y[n] for n,i in enumerate(fn(x,*b)) if y[n]>0])/float(len(y)-len(b))
	def guess_A0(self):
		print 'Fitting Peaks in',self.filename
		A0,gammas = [],[]
		a,b,L = self.calibration[0],self.calibration[1],len(self.spec)
		for itp in self.isotope_list:
			ip = isotope(itp)
			gammas += [(ip.name(),ip.decay_const())+g for g in ip.gammas(Imin=0.11) if int((g[0]-b)/a)<L]
		clip = [int(i-self.SNP[n]) if int(i-self.SNP[n])>max((1.5,1.5*np.sqrt(self.SNP[n]))) else 0 for n,i in enumerate(self.exp_smooth(self.spec,alpha=0.35))]
		gammas = [g+(int((g[2]-b)/a),) for g in gammas if clip[int((g[2]-b)/a)]>0 and abs(g[2]-511.0)>10 and int((g[2]-b)/a)>0]
		N_c = [int(2.506*self.resolution*g[-1]**0.5*clip[g[-1]])+int(2.0*self.pk_args[0]*self.pk_args[1]*self.resolution*g[-1]**0.5*clip[g[-1]]*np.exp(-0.5/self.pk_args[1]**2)) for g in gammas]
		A0 = [N_c[n]/(g[3]*self.get_efficiency(g[2])) for n,g in enumerate(gammas)]
		itps = {i:[] for i in list(set([g[0] for g in gammas]))}
		for n,g in enumerate(gammas):
			itps[g[0]].append((A0[n],g[3],N_c[n]))
		return {itp:round(np.average([i[0] for i in a],weights=[np.sqrt(i[2]*(i[0]/i[2])**2+i[1]**2) for i in a]),0) for itp,a in itps.iteritems()}
	def get_p0(self,A0=None):
		if A0 is None:
			A0 = self.guess_A0()
		if len(A0)==0:
			return []
		gammas = []
		a,b,L,RS = self.calibration[0],self.calibration[1],len(self.spec),self.resolution
		for itp,A in A0.iteritems():
			ip = isotope(itp)
			gammas += [(itp,A,ip.decay_const(),((g[0]-b)/a))+g for g in ip.gammas(Imin=0.11) if ((g[0]-b)/a)<L and abs(g[0]-511.0)>10]
		N_c = [g[1]*(self.get_efficiency(g[4])*g[5]) for g in gammas]
		A = [int(N_c[n]/(2.506*RS*g[3]**0.5+2*self.pk_args[0]*self.pk_args[1]*RS*g[3]**0.5*np.exp(0.5/self.pk_args[1]**2))) for n,g in enumerate(gammas)]
		gammas = sorted([g+(int(g[3]-6.0*RS*g[3]**0.5),int(g[3]+5.5*RS*g[3]**0.5),A[n]) for n,g in enumerate(gammas) if A[n]>max((1.5,1.5*np.sqrt(self.SNP[int(g[3])])))],key=lambda h:h[4])
		pks = [[gammas[0]]]
		for g in gammas[1:]:
			if g[7]<pks[-1][-1][8]:
				pks[-1].append(g)
			else:
				pks.append([g])
		p0 = []
		for p in pks:
			m,b = self.simple_regression(range(p[0][7],p[-1][8]),self.SNP[p[0][7]:p[-1][8]])
			bm = max((self.SNP[int(p[0][3])]**0.5,1.0))
			p0.append({'l':p[0][7],'h':p[-1][8]+1,'p0':[m,b],'pk_info':[],'bounds':([m-1e-5,b-1.5*bm],[m+1e-5,b+1.5*bm])})
			for gm in p:
				p0[-1]['pk_info'].append({'istp':gm[0],'eng':gm[4],'I':gm[5],'unc_I':gm[6],'lm':gm[2]})
				p0[-1]['p0'] += [gm[9],gm[3],RS*gm[3]**0.5]
				p0[-1]['bounds'] = (p0[-1]['bounds'][0]+[0,gm[3]-0.25*RS*gm[3]**0.5-2.0,0.75*RS*gm[3]**0.5],p0[-1]['bounds'][1]+[7.5*abs(gm[9])+1,gm[3]+0.25*RS*gm[3]**0.5+2.0,1.5*RS*gm[3]**0.5])
		return p0
	def filter_fits(self,fits,loop=True):
		good_fits = []
		for f in fits:
			N = (len(f['fit'])-2)/3
			chi2 = self.chi2(self.Npeak,range(f['l'],f['h']),f['fit'],self.spec[f['l']:f['h']])
			for n,p in enumerate(f['pk_info']):
				ft,u = f['fit'],f['unc']
				p['N'] = int(ft[2+3*n]*(2.506*ft[4+3*n]+2*self.pk_args[0]*self.pk_args[1]*ft[4+3*n]*np.exp(-0.5/self.pk_args[1]**2)))
				p['unc_N'] = int(min((np.sqrt(abs(u[2+3*n][2+3*n])*(p['N']/ft[2+3*n])**2+abs(u[4+3*n][4+3*n])*(p['N']/ft[4+3*n])**2),1e18)))
			keep = [n for n in range(N) if f['fit'][2+3*n]>max((2.5,2.5*np.sqrt(self.SNP[int(f['fit'][3+3*n])]))) and (f['pk_info'][n]['unc_N']<25*f['pk_info'][n]['N']) and chi2<500.0]
			for n in keep:
				arr = [0,1,2+3*n,3+3*n,4+3*n]
				good_fits.append({'fit':[f['fit'][i] for i in arr]})
				good_fits[-1]['unc'] = [[f['unc'][i][j] for j in arr] for i in arr]
				good_fits[-1]['p0'] = [f['p0'][i] for i in arr]
				good_fits[-1]['pk_info'] = [f['pk_info'][n]]
				good_fits[-1]['bounds'] = ([f['bounds'][0][i] for i in arr],[f['bounds'][1][i] for i in arr])
				good_fits[-1]['l'],good_fits[-1]['h'] = int(f['fit'][3+3*n]-5.0*f['fit'][4+3*n]-3),int(f['fit'][3+3*n]+4.5*f['fit'][4+3*n]+3)
		good_fits = sorted(good_fits,key=lambda h:h['fit'][3])
		if len(good_fits)==0:
			self.peak_fits = []
			return []
		groups = [good_fits[0]]
		for g in good_fits[1:]:
			if g['l']<groups[-1]['h']:
				L = len(groups[-1]['fit'])
				groups[-1]['fit'] += g['fit'][2:]
				groups[-1]['unc'] = [gp+[0,0,0] for gp in groups[-1]['unc']]+[[0 for i in range(L)]+gp[2:] for gp in g['unc'][2:]]
				groups[-1]['p0'] += g['p0'][2:]
				groups[-1]['pk_info'].append(g['pk_info'][0])
				groups[-1]['bounds'] = (groups[-1]['bounds'][0]+g['bounds'][0][2:],groups[-1]['bounds'][1]+g['bounds'][1][2:])
				groups[-1]['h'] = g['h']
			else:
				groups.append(g)
		for g in groups:
			chi2 = self.chi2(self.Npeak,range(g['l'],g['h']),g['fit'],self.spec[g['l']:g['h']])
			for n,p in enumerate(g['pk_info']):
				p['chi2'] = chi2
		self.peak_fits = None if loop else groups
		return self.filter_fits(groups,loop=False) if loop else groups
	def calc_A0(self,fits):
		gammas = []
		for f in fits:
			gammas += f['pk_info']
		A0 = [g['N']/(g['I']*self.get_efficiency(g['eng'])) for g in gammas]
		itps = {i:[] for i in list(set([g['istp'] for g in gammas]))}
		for n,g in enumerate(gammas):
			itps[g['istp']].append((A0[n],np.sqrt(g['unc_N']**2*(A0[n]/g['N'])**2+g['unc_I']**2*(A0[n]/g['I'])**2)))
		return {itp:np.average([i[0] for i in a],weights=[1.0/i[1]**2 for i in a]) for itp,a in itps.iteritems()}
	def fit_peaks(self,A0=None):
		if A0 is None and self.peak_fits is not None:
			return self.peak_fits
		p0 = self.get_p0(A0)
		fits = []
		for p in p0:
			try:
				p['fit'],p['unc'] = curve_fit(self.Npeak,range(p['l'],p['h']),self.spec[p['l']:p['h']],p0=p['p0'],bounds=p['bounds'],sigma=np.sqrt(self.SNP[p['l']:p['h']]))
				fits.append(p)
			except Exception, err:
				print 'Error on peak:',p['pk_info']
				print Exception, err
		if A0 is None:
			return self.fit_peaks(A0=self.calc_A0(self.filter_fits(fits)))
		return self.filter_fits(fits)
	def plot_fits(self,logscale=True,wfit=False,bg=False,save=False,zoom=False,subpeak=False,printout=False):
		f,ax = plt.subplots() if not save else plt.subplots(figsize=(12.8,4.8))
		mn,mx = 0,len(self.spec)
		if wfit:
			for n,pk in enumerate(self.fit_peaks()):
				if printout:
					print pk['pk_info']
				if zoom:
					mn,mx = pk['l']-20 if mn==0 else mn,pk['h']+20
				a,b = self.calibration[0],self.calibration[1]
				ax.plot([a*i+b for i in np.arange(pk['l'],pk['h'],0.1)],self.Npeak(np.arange(pk['l'],pk['h'],0.1),*pk['fit']),lw=1.2,color=self.pallate['r'],zorder=10,label=('Peak Fit'+('s' if len(self.peak_fits)>1 else '') if n==0 else None))
				if subpeak:
					N = (len(pk['fit'])-2)/3
					if N>1:
						for m in range(N):
							p = pk['fit'][:2]+pk['fit'][2+m*3:5+m*3]
							ax.plot([a*i+b for i in np.arange(pk['l'],pk['h'],0.1)],self.Npeak(np.arange(pk['l'],pk['h'],0.1),*p),lw=1.0,ls='--',color=self.pallate['r'],zorder=5)
		if bg:
			ax.plot(self.get_energy()[mn:mx],self.SNP[mn:mx],lw=1.0,color=self.pallate['b'],label='Background',zorder=3)
		ax.plot(self.get_plot_energy(mn,mx),self.get_plot_spec(logscale,mn,mx),lw=1.0,color=self.pallate['k'],label=self.filename,zorder=1)
		if logscale:
			ax.set_yscale('log')
		ax.set_ylim((max((0.9,ax.get_ylim()[0])),ax.get_ylim()[1]))
		ax.set_xlabel('Energy [keV]')
		ax.set_ylabel('Counts')
		ax.legend(loc=0)
		f.tight_layout()
		if save:
			f.savefig('../plots/'+self.directory+'/peak_fits/'+self.filename+'.png')
			f.savefig('../plots/'+self.directory+'/peak_fits/'+self.filename+'.pdf')
			plt.close()
		else:
			plt.show()



class calibration(object):
	def __init__(self,directory):
		self.db = _db
		self.db_connection = _db_connection
		self.directory = directory
		fnms = [str(i[0]) for i in self.db.execute('SELECT * FROM calibration_files WHERE directory=?',(self.directory,))]
		self.spectra = [spectrum(s) for s in fnms]
		self.pallate = {'k':'#2c3e50','b':'#2980b9','r':'#c0392b','y':'#f39c12','p':'#8e44ad','g':'#27ae60','gy':'#7f8c8d','o':'#d35400','w':'#ecf0f1','aq':'#16a085'}
	def objective(self,m,b):
		L = len(self.log_peaks[0])
		I = [[int((e-b)/m) for e in eng] for eng in self.calib_peak_eng]
		return sum([sum([s[i] for i in I[n] if 0<i<L]) for n,s in enumerate(self.log_peaks)])
	def guess_energy_cal(self):
		guess = [[float(m) for m in i[1].split(',')] for i in self.db.execute('SELECT * FROM calibration WHERE directory=?',(self.directory,))][0]
		delta0 = (0.05*guess[0],10.0)
		gammas = [isotope(s.isotope_list[0]).gammas(Imin=0.3,Emin=70) for s in self.spectra]
		self.calib_peak_eng = [[g[0] for g in gm] for gm in gammas]
		self.log_peaks = [[np.log(np.log(np.sqrt(max((i-s.SNP[n],0))+1.0)+1.0)+1.0) for n,i in enumerate(s.spec)] for s in self.spectra]
		K = 200
		sm = sum([1.0/float(i)**2 for i in range(1,K+1)])
		wts = [1.0/(sm*float(i)**2) for i in range(1,K+1)]
		best_guess = (self.objective(guess[0],guess[1]),guess)
		for i in range(10):
			p0 = [(guess[0]+delta0[0]*g*np.exp(-i/4.0),guess[1]+delta0[1]*g*np.exp(-i/4.0)) for g in np.random.normal(0,1,K)]
			G = sorted([(self.objective(p[0],p[1]),p) for p in p0],key=lambda h:h[0],reverse=True)
			if G[0][0]>best_guess[0]:
				best_guess = G[0]
			guess = (sum([g[1][0]*wts[n] for n,g in enumerate(G)]),sum([g[1][1]*wts[n] for n,g in enumerate(G)]))
		return best_guess[1]
	def fits(self,usecal=True):
		guess = self.spectra[0].calibration if usecal else self.guess_energy_cal()
		self.db.execute('DELETE FROM calibration_peaks WHERE directory=?',(self.directory,))
		for s in self.spectra:
			s.set_calibration([guess[0],guess[1]])
			for pk in s.fit_peaks():
				N = (len(pk['p0'])-2)/3
				for n in range(N):
					w = 'pk_info'
					pkinfo = [s.filename+'.Spe',self.directory,pk[w][n]['istp'],pk[w][n]['N'],pk[w][n]['unc_N'],pk[w][n]['chi2'],pk[w][n]['eng'],pk[w][n]['I'],pk[w][n]['unc_I'],pk['fit'][3+n*3],np.sqrt(pk['unc'][3+n*3][3+n*3])]
					pkinfo += [pk['fit'][4+n*3],np.sqrt(pk['unc'][4+n*3][4+n*3])]
					self.db.execute('INSERT INTO calibration_peaks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',tuple(pkinfo))
			s.plot_fits(wfit=True,logscale=True,save=True,zoom=True)
			self.db_connection.commit()
	def energy_calibration(self,show=False):
		guess = self.spectra[0].calibration
		pks = [[float(i[0]),float(i[1]),float(i[2])] for i in self.db.execute('SELECT energy,mu,unc_mu FROM calibration_peaks WHERE directory=? ORDER BY "energy" ASC',(self.directory,)) if float(i[2])<float(i[1])]
		fit,unc = curve_fit(lambda x,m,b:[m*i+b for i in x],[p[1] for p in pks],[p[0] for p in pks],p0=[guess[0],guess[1]],sigma=[guess[0]*p[2] for p in pks])
		self.db.execute('UPDATE calibration SET energy_calibration=? WHERE directory=?',(','.join([str(i) for i in fit]),self.directory))
		self.db_connection.commit()
		f,ax = plt.subplots(2,sharex=True,gridspec_kw = {'height_ratios':[3, 2]})
		xbins = np.arange(pks[0][1]-40,pks[-1][1]+20,0.5)
		ax[0].plot(xbins,[fit[0]*i+fit[1] for i in xbins],lw=2.0,color=self.pallate['gy'],label=r'$E_{\gamma}='+str(round(fit[0],3))+r'\cdot N'+('+' if fit[1]>0 else '')+str(round(fit[1],3))+r'$')
		ax[0].errorbar([p[1] for p in pks],[p[0] for p in pks],yerr=[fit[0]*p[2] for p in pks],color=self.pallate['k'],ls='None',marker='o',label='Peak Centroids')
		ax[0].set_ylabel('Energy [keV]')
		ax[0].legend(loc=0)
		ax[1].plot(xbins,[0.0 for i in xbins],ls='--',color=self.pallate['gy'],lw=2.0)
		label = r'$\sigma_E='+str(round(np.std([p[0]-fit[0]*p[1]-fit[1] for p in pks]),3))+r'$ [keV], $\chi^2_{\nu}='+str(round(sum([(p[0]-fit[0]*p[1]-fit[1])**2/p[2]**2 for p in pks])/float(len(pks)-3),3))+r'$'
		ax[1].errorbar([p[1] for p in pks],[p[0]-fit[0]*p[1]-fit[1] for p in pks],yerr=[fit[0]*p[2] for p in pks],color=self.pallate['k'],ls='None',marker='o',label=label)
		ax[1].set_xlabel('Channel Number')
		ax[1].set_ylabel('Fit Residual [keV]')
		ax[1].legend(loc=0)
		f.tight_layout()
		f.subplots_adjust(hspace=0.1)
		if show:
			plt.show()
		else:
			f.savefig('../plots/'+self.directory+'/calibration/energy_calibration.png')
			f.savefig('../plots/'+self.directory+'/calibration/energy_calibration.pdf')
			plt.close()
	def get_efficiency(self,energy,efficiency):
		a,b,c = efficiency[0],efficiency[1],efficiency[2]
		return np.exp(a*np.log(energy)**2+b*np.log(energy)+c)
	def parse_peak_dat(self,l):
		return {'fnm':str(l[0]),'dir':str(l[1]),'istp':str(l[2]),'N':int(l[3]),'unc_N':int(l[4]),'chi2':float(l[5]),'eng':float(l[6]),'I':float(l[7]),'unc_I':float(l[8])}
	def efficiency_calibration(self,show=False):
		guess,A0,T0 = [0.566635,-8.5808,27.56425],{},{}
		for c in [[str(i[0]),float(i[3]),[[int(t) for t in m.split(':')] for m in str(i[4]).split('::')]] for i in self.db.execute('SELECT * FROM calibration_files WHERE directory=?',(self.directory,))]:
			A0[c[0]],T0[c[0]] = c[1],dtm.datetime(c[2][0][0],c[2][0][1],c[2][0][2],c[2][1][0],c[2][1][1],c[2][1][2])
		f,ax = plt.subplots()
		E_exp,N_eff,sig_N_eff = [],[],[]
		for pk in [self.parse_peak_dat(i) for i in self.db.execute('SELECT * FROM calibration_peaks WHERE directory=?',(self.directory,)) if float(i[5])<300]:
			if pk['I']>=0.5 and pk['eng']>=50 and pk['unc_N']<pk['N'] and pk['fnm'] in [i.filename+'.Spe' for i in self.spectra]:
				lm = isotope(pk['istp']).decay_const()
				fl = [i for i in self.spectra if i.filename+'.Spe'==pk['fnm']][0]
				conv = lm/(A0[pk['fnm']]*np.exp(-lm*(fl.start_time-T0[pk['fnm']]).total_seconds())*(1.0-np.exp(-lm*fl.live_time)))
				E_exp.append(pk['eng'])
				N_eff.append(pk['N']*conv/(0.01*pk['I']))
				sig_N_eff.append(pk['unc_N']*conv/(0.01*pk['I']))
		efit,unc = curve_fit(lambda x,a,b,c:[np.exp(a*np.log(i)**2+b*np.log(i)+c) for i in x],E_exp,N_eff,p0=guess,sigma=sig_N_eff)		
		lbl = r'$\epsilon(E)=exp['+str(round(efit[0],3))+r'\cdot ln(E)^2'+('+' if efit[1]>0 else '')+str(round(efit[1],1))+r'\cdot ln(E)'+('+' if efit[2]>0 else '')+str(round(efit[2],1))+r']$'
		lbl2 = r'Peak Fits, $\chi^2_{\nu}='+str(round(sum([(N_eff[n]-(np.exp(efit[0]*np.log(i)**2+efit[1]*np.log(i)+efit[2])))**2/sig_N_eff[n]**2 for n,i in enumerate(E_exp)])/float(len(E_exp)-4),3))+r'$'
		ax.plot(range(int(min(E_exp))-10,int(max(E_exp))+20),[np.exp(efit[0]*np.log(i)**2+efit[1]*np.log(i)+efit[2]) for i in range(int(min(E_exp))-10,int(max(E_exp))+20)],ls='-',lw=2.0,color=self.pallate['k'],label=lbl)
		ax.errorbar(E_exp,N_eff,yerr=sig_N_eff,ls='None',marker='o',color=self.pallate['gy'],label=lbl2)
		self.db.execute('UPDATE calibration SET efficiency_calibration=? WHERE directory=?',(','.join([str(i) for i in efit]),self.directory))
		self.db_connection.commit()
		ax.set_yscale('log')
		ax.set_xlabel('Energy [keV]')
		ax.set_ylabel('Efficiency [a.u.]')
		ax.legend(loc=0)
		f.tight_layout()
		if show:
			plt.show()
		else:
			f.savefig('../plots/'+self.directory+'/calibration/efficiency_calibration.pdf')
			f.savefig('../plots/'+self.directory+'/calibration/efficiency_calibration.png')
			plt.close()
	def resolution_calibration(self,show=False):
		f,ax = plt.subplots()
		guess = [0.05]
		pks = [[float(i[0]),float(i[1]),float(i[2]),float(i[3])] for i in self.db.execute('SELECT mu,unc_mu,sig,unc_sig FROM calibration_peaks WHERE directory=?',(self.directory,)) if float(i[1])<float(i[0])]
		ax.errorbar([np.sqrt(p[0]) for p in pks],[p[2] for p in pks],yerr=[np.sqrt((guess[0]*0.5*p[1]/np.sqrt(p[0]))**2+p[3]**2) for p in pks],ls='None',marker='o',color=self.pallate['r'])
		fit,unc = curve_fit(lambda x,r:[r*i for i in x],[np.sqrt(p[0]) for p in pks],[p[2] for p in pks],p0=guess,sigma=[np.sqrt((guess[0]*0.5*p[1]/np.sqrt(p[0]))**2+p[3]**2) for p in pks])
		self.db.execute('UPDATE calibration SET resolution=? WHERE directory=?',(fit[0],self.directory))
		self.db_connection.commit()
		mu_range = np.arange(min([np.sqrt(p[0]) for p in pks])-4,max([np.sqrt(p[0]) for p in pks])+4,1)
		ax.plot(mu_range,[fit[0]*i for i in mu_range],lw=2.0,color=self.pallate['k'],label=r'$\sigma='+str(round(fit[0],3))+r'\sqrt{bin}$')
		ax.set_xlabel(r'$\sqrt{Bin}$')
		ax.set_ylabel(r'Peak Width $\sigma$ [a.u.]')
		ax.legend(loc=0)
		f.tight_layout()
		if show:
			plt.show()
		else:
			f.savefig('../plots/'+self.directory+'/calibration/resolution_calibration.pdf')
			f.savefig('../plots/'+self.directory+'/calibration/resolution_calibration.png')
			plt.close()

class experiment(object):
	def __init__(self,directory):
		self.directory = directory
		self.db_connection = _db_connection
		self.db = _db
		self.pallate = {'k':'#2c3e50','b':'#2980b9','r':'#c0392b','y':'#f39c12','p':'#8e44ad','g':'#27ae60','gy':'#7f8c8d','o':'#d35400','w':'#ecf0f1','aq':'#16a085'}
	def update_files(self):
		for table in ['experiment_files','calibration_files','irradiation_history','calibration']:
			self.db.execute('DELETE FROM '+table+' WHERE directory=?',(self.directory,))
		mass = [i for i in open('../data/spectra/'+self.directory+'/mass.txt').read().split('*') if i!='']
		expt,cal,hist = [l for l in mass[0].split('\n') if l!=''],[l for l in mass[1].split('\n') if l!=''],[l for l in mass[2].split('\n') if l!='']
		types = {'f':str,'m':float,'i':str,'E0_D2':float,'R_smpl':float,'R_src':float,'dx':float,'dy':float,'dz':float,'elbow':int,'c':str,'a0':float,'d':str,'start':str,'stop':str}
		for ln in expt:
			m = {i.split('=')[0]:types[i.split('=')[0]](i.split('=')[1]) for i in ln.split(',')}
			self.db.execute('INSERT INTO experiment_files VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(m['f'],self.directory,m['m'],m['i'],m['E0_D2'],m['dx'],m['dy'],m['dz'],m['R_src'],m['R_smpl'],m['elbow'],0.0,0.0))
		for ln in cal:
			m = {i.split('=')[0]:types[i.split('=')[0]](i.split('=')[1]) for i in ln.split(',')}
			self.db.execute('INSERT INTO calibration_files VALUES(?,?,?,?,?)',(m['c'],self.directory,m['i'],m['a0'],m['d']))
		f = open('../data/spectra/'+self.directory+'/'+m['c'],'r').read().split('\n')[-5]
		eng_cal = [float(f.split(' ')[1]),float(f.split(' ')[0])]
		resolution = 0.04
		efficiency = [0.566635,-8.5808,27.56425]
		pk_args = (0.31,0.91)
		self.db.execute('INSERT INTO calibration VALUES(?,?,?,?,?,?)',(self.directory,','.join([str(i) for i in eng_cal]),','.join([str(i) for i in efficiency]),resolution,pk_args[0],pk_args[1]))
		for ln in hist:
			m = {i.split('=')[0]:types[i.split('=')[0]](i.split('=')[1]) for i in ln.split(',')}
			self.db.execute('INSERT INTO irradiation_history VALUES(?,?,?)',(self.directory,m['start'],m['stop']))
		self.db_connection.commit()
	def create_plots_dir(self):
		os.chdir('../plots')
		os.system('mkdir '+self.directory)
		os.chdir(self.directory)
		for dr in ['calibration','decay_curves','peak_fits','cross_sections','monitors']:
			os.system('mkdir '+dr)
		os.system('cp -a ../cross_sections/. cross_sections/')
		os.chdir('../../reports')
		os.system('mkdir '+self.directory)
		os.system('cp template/template.tex '+self.directory+'/'+self.directory+'_report.tex')
		os.chdir('../code')
	def fit_experiment_peaks(self):
		eob = [[[int(t) for t in m.split(':')] for m in i[2].split('::')] for i in self.db.execute('SELECT * FROM irradiation_history WHERE directory=? ORDER BY stop_time DESC',(self.directory,))][0]
		eob = dtm.datetime(eob[0][0],eob[0][1],eob[0][2],eob[1][0],eob[1][1],eob[1][2])
		for fnm in [str(i[0]) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]:
			sp = spectrum(fnm)
			self.db.execute('DELETE FROM experiment_peaks WHERE filename=?',(fnm,))
			for pk in sp.fit_peaks():
				N = (len(pk['p0'])-2)/3
				for n in range(N):
					w = 'pk_info'
					pkinfo = [sp.filename+'.Spe',pk[w][n]['istp'],pk[w][n]['eng'],pk[w][n]['I'],pk[w][n]['unc_I'],pk[w][n]['N'],pk[w][n]['unc_N'],pk[w][n]['chi2'],isotope(pk[w][n]['istp']).decay_const()]
					pkinfo += [sp.get_efficiency(pk[w][n]['eng']),sp.live_time,(sp.start_time-eob).total_seconds()]
					if pk[w][n]['unc_N']/pk[w][n]['N']<0.02:
						self.db.execute('INSERT INTO experiment_peaks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',tuple(pkinfo))
			sp.plot_fits(wfit=True,logscale=True,save=True,zoom=True)
			self.db_connection.commit()
	def update_calibration(self):
		cb = calibration(self.directory)
		for i in range(2):
			cb.fits(usecal=(False if i==0 else True))
			cb.energy_calibration()
			cb.resolution_calibration()
		cb.efficiency_calibration()
	def parse_calibration_peaks(self,i):
		return {'fnm':str(i[0]),'istp':str(i[2]),'N':int(i[3]),'unc_N':int(i[4]),'chi2':float(i[5]),'eng':float(i[6]),'I':float(i[7]),'unc_I':float(i[7])}
	def parse_experiment_peaks(self,i):
		return {'fnm':str(i[0]),'istp':str(i[1]),'eng':float(i[2]),'I':float(i[3]),'unc_I':float(i[4]),'N':float(i[5]),'unc_N':float(i[6]),'chi2':float(i[7]),'lm':float(i[8]),'eff':float(i[9]),'t_m':float(i[10]),'t_c':float(i[11])}
	def get_A0(self,fnm,istp,saveplot=True,show=False):
		ip = isotope(istp)
		lm = ip.decay_const()
		pks = [self.parse_experiment_peaks(i) for i in self.db.execute('SELECT * FROM experiment_peaks WHERE filename=? AND isotope=?',(fnm,istp))]
		A = [p['N']*lm/((1.0-np.exp(-lm*p['t_m']))*p['eff']*0.01*p['I']) for p in pks]
		T_c = [p['t_c'] for t in pks]
		sigA = [np.sqrt(p['unc_N']**2*(A[n]/p['N'])**2+p['unc_I']**2*(A[n]/p['I'])**2) for n,p in enumerate(pks)]
		fit,unc = curve_fit(lambda x,a0:a0*np.exp(-lm*x),T_c,A,sigma=sigA,p0=[A[0]*np.exp(lm*pks[0]['t_c'])])
		unc[0][0] = np.average([sigA[n]*np.exp(lm*p['t_c']) for n,p in enumerate(pks)],weights=[1.0/(sigA[n]*np.exp(lm*p['t_c']))**2 for n,p in enumerate(pks)])**2
		if saveplot or show:
			f,ax = plt.subplots()
			ax.errorbar([(1.0/60.0)*t for t in T_c],[a for a in A],yerr=[a for a in sigA],ls='None',marker='o',capsize=5.0,color=self.pallate['gy'],label='Photopeak Activity')
			T = np.arange(0,1.05*max(T_c),0.01*max(T_c))
			ax.plot((1.0/60.0)*T,[fit[0]*np.exp(-lm*t) for t in T],lw=2.0,color=self.pallate['k'],label='Exponential Fit')
			ax.plot((1.0/60.0)*T,[(fit[0]*np.exp(-lm*t)+np.sqrt(unc[0][0])*np.exp(-lm*t)) for t in T],ls='--',lw=2.0,color=self.pallate['k'],label=r'$\pm 1\sigma_{fit}$')
			ax.plot((1.0/60.0)*T,[(fit[0]*np.exp(-lm*t)-np.sqrt(unc[0][0])*np.exp(-lm*t)) for t in T],ls='--',lw=2.0,color=self.pallate['k'])
			ax.set_xlabel('Cooling Time [min]')
			ax.set_ylabel('Activity [kBq]')
			ax.set_title(ip.TeX()+' Activity')
			ax.legend(loc=0)
			f.tight_layout()
			if saveplot:
				f.savefig('../plots/'+self.directory+'/decay_curves/'+fnm.split('.')[0]+'_'+istp+'.png')
				f.savefig('../plots/'+self.directory+'/decay_curves/'+fnm.split('.')[0]+'_'+istp+'.pdf')
			if show:
				plt.show()
			else:
				plt.close()
		return fit[0],np.sqrt(unc[0][0])
	def parse_experiment_file(self,i):
		return {'f':str(i[0]),'m':float(i[2]),'i':str(i[3]).split(';'),'E0_D2':float(i[4]),'dx':float(i[5]),'dy':float(i[6]),'dz':float(i[7]),'R_src':float(i[8]),'R_smpl':float(i[9]),'elbow':bool(int(i[10]))}
	def calculate_flux(self,saveplot=True,show=False):
		flx = flux_calc(self.directory)
		mass = {'115INm':{'ab':0.957,'M':114.903},'116INm':{'ab':0.957,'M':114.903},'113INm':{'ab':0.042,'M':112.904},'58CO':{'ab':0.68077,'M':57.935}}
		reac = {'115INm':'115IN_n_inl_115INm','116INm':'115IN_n_g_116INm','113INm':'113IN_n_inl_113INm','58CO':'58NI_n_p_58CO'}
		self.flux_detailed = {}
		for fl in [self.parse_experiment_file(i) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]:
			flux = []
			self.flux_detailed[fl['f']] = {}
			for istp in fl['i']:
				A0,sig_A0 = self.get_A0(fl['f'],istp)
				flx.set_sample_params(E0_D2=fl['E0_D2'],R_smpl=fl['R_smpl'],R_src=fl['R_src'],dx=fl['dx'],dy=fl['dy'],dz=fl['dz'],elbow=fl['elbow'])
				flx.plot_energy_spectrum(saveplot=True,show=False)
				flx.plot_intensity_angle(saveplot=True,show=False)
				flx.plot_energy_angle(saveplot=True,show=False)
				sig,unc_sig = flx.get_average_xs(reaction=reac[istp])
				# print 'Monitor XS=',sig,'+/-',unc_sig
				I_n = self.one_step_batemann(istp=istp)
				phi = A0/(1e-3*sig*(mass[istp]['ab']*fl['m']*6.022E-1/mass[istp]['M'])*I_n)
				unc_phi = np.sqrt(sig_A0**2*(phi/A0)**2+unc_sig**2*(phi/sig)**2)
				self.flux_detailed[fl['f']][istp] = {'sig':sig,'unc_sig':unc_sig,'E':flx.EdE[0],'dE':flx.EdE[1],'phi':phi,'unc_phi':unc_phi}
				flux.append((istp,phi,unc_phi))
			self.db.execute('UPDATE experiment_files SET flux=?, unc_flux=? WHERE filename=?',(np.average([i[1] for i in flux],weights=[1.0/i[2]**2 for i in flux]),np.sqrt(np.average([i[2]**2 for i in flux],weights=[1.0/i[2]**2 for i in flux])),fl['f']))
		self.db_connection.commit()
	def one_step_batemann(self,istp='115INm',units='hours',saveplot=True,show=False):
		u_conv = {'seconds':1.0,'minutes':60.0,'hours':3600.0,'days':24.0*3600.0,'weeks':7.0*24.0*3600.0}[units]
		ip = isotope(istp)
		lm = ip.decay_const()
		start = [[[int(t) for t in m.split(':')] for m in i[1].split('::')] for i in self.db.execute('SELECT * FROM irradiation_history WHERE directory=? ORDER BY stop_time ASC',(self.directory,))]
		start = [dtm.datetime(t[0][0],t[0][1],t[0][2],t[1][0],t[1][1],t[1][2]) for t in start]
		stop = [[[int(t) for t in m.split(':')] for m in i[2].split('::')] for i in self.db.execute('SELECT * FROM irradiation_history WHERE directory=? ORDER BY stop_time ASC',(self.directory,))]
		stop = [dtm.datetime(t[0][0],t[0][1],t[0][2],t[1][0],t[1][1],t[1][2]) for t in stop]
		I_n = [1.0-np.exp(-lm*(stop[0]-start[0]).total_seconds())]
		N = len(start)
		# for n in range(N):
		# 	print start[n].strftime("%H:%M %m/%d/%Y"),'&',stop[n].strftime("%H:%M %m/%d/%Y"),'&',round((stop[n]-start[n]).total_seconds()/3600.0,1),r'\\'
		for n in range(1,N):
			I_n.append(1.0-(1.0-I_n[-1]*np.exp(-lm*(stop[n]-stop[n-1]).total_seconds()))*np.exp(-lm*(stop[n]-start[n]).total_seconds()))
		if saveplot or show:
			dT = 0.05/lm
			T = np.arange(0,(stop[0]-start[0]).total_seconds(),dT).tolist()
			I = [(1.0-np.exp(-lm*t)) for t in T]
			for m in range(1,N):
				T_m = np.arange((stop[m-1]-start[0]).total_seconds(),(start[m]-start[0]).total_seconds(),dT).tolist()
				I_m = [I_n[m-1]*np.exp(-lm*(t-(stop[m-1]-start[0]).total_seconds())) for t in T_m]
				T += T_m
				I += I_m
				T_m = np.arange((start[m]-start[0]).total_seconds(),(stop[m]-start[0]).total_seconds(),dT).tolist()
				I_m = [1.0-(1.0-I_n[m-1]*np.exp(-lm*(t-(stop[m-1]-start[0]).total_seconds())))*np.exp(-lm*(t-(start[m]-start[0]).total_seconds())) for t in T_m]
				T += T_m
				I += I_m
			T = [i/u_conv for i in T]
			f,ax = plt.subplots()
			ax.plot(T,I,color=self.pallate['k'],lw=2.0,label='Saturation Fraction')
			ax.set_xlabel('Experiment Time ['+units+']')
			ax.set_ylabel('Saturation Fraction [a.u.]')
			ax.legend(loc=0)
			f.tight_layout()
			if saveplot:
				f.savefig('../plots/'+self.directory+'/monitors/'+istp+'_batemann.png')
				f.savefig('../plots/'+self.directory+'/monitors/'+istp+'_batemann.pdf')
			if show:
				plt.show()
			else:
				plt.close()
		return I_n[-1]
	def flux_TeX(self,flux,unc_flux):
		N = int(np.log(flux)/np.log(10))
		return '('+str(round(flux/10**N,2))+r'$\pm$'+str(round(unc_flux/10**N,2))+r')$\cdot 10^'+str(N)+r'$'
	def save_tex(self,ss,tex):
		f = open(tex+'.tex','w')
		f.write(ss)
		f.close()
	def generate_abstract(self):
		ab = 'In this experiment we use the foil activation technique to measure the fast neutron flux in the Berkeley High Flux Neutron Generator (HFNG).  '
		ab += 'The irradiation began at '
		times = [[[m.split(':') for m in str(i[1]).split('::')],[m.split(':') for m in str(i[2]).split('::')]] for i in self.db.execute('SELECT * FROM irradiation_history WHERE directory=? ORDER BY stop_time ASC',(self.directory,))]
		ab += times[0][0][1][0]+':'+times[0][0][1][1]+' on '+times[0][0][0][1]+'/'+times[0][0][0][2]+'/'+times[0][0][0][0]
		ab += ' and ended at '+times[-1][1][1][0]+':'+times[-1][1][1][1]+' on '+times[-1][1][0][1]+'/'+times[-1][1][0][2]+'/'+times[-1][1][0][0]
		flx = sorted([(float(i[11]),float(i[12])) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))],key=lambda h:h[0])
		ab += r'  The peak flux measured was '+self.flux_TeX(flx[-1][0],flx[-1][1])+r' [$\frac{n}{cm^2s}$] using the following monitor channel(s): '
		chnl = {'115INm':r'$^{115}$In(n,n'+"'"+r')$^{115m}$In','113INm':r'$^{113}$In(n,n'+"'"+r')$^{113m}$In','116INm':r'$^{115}$In(n,$\gamma$)$^{116m}$In','58CO':r'$^{58}$Ni(n,p)$^{58}$Co'}
		ab += ', '.join([chnl[m] for m in [str(i[3]).split(';') for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))][0]])+'.'
		self.save_tex(ab,'abstract')
	def generate_batemann(self):
		ss = ''
		for istp in [str(i[3]).split(';') for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))][0]:
			ss += r'\begin{figure}[htb]'+'\n'
			ss += r'\includegraphics[width=9cm]{monitors/'+istp+'_batemann.pdf}'+'\n'
			ss += r'\caption{Saturation fraction, $F_S$ for the '+isotope(istp).TeX()+' isotope during the irradiation period.\n}\n'
			ss += r'\label{fig:'+istp+'_batemann}'+'\n'
			ss += r'\end{figure}'+'\n\n'
		self.save_tex(ss,'batemann')
	def generate_decay_curves(self):
		ss = ''
		for smpl,fnm in enumerate([str(i[0]) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]):
			for istp in [str(i[3]).split(';') for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=? AND filename=?',(self.directory,fnm))][0]:
				ss += r'\begin{figure}[htb]'+'\n'
				ss += r'\includegraphics[width=9cm]{decay_curves/'+fnm.split('.')[0]+'_'+istp+'.pdf}\n'
				ss += r'\caption{Exponential fit to the peak activities of '+isotope(istp).TeX()+' measured in Sample '+str(smpl)+'.\n}\n'
				ss += r'\label{fig:'+fnm.split('.')[0]+'_'+istp+'}\n'
				ss += r'\end{figure}'+'\n\n'
		self.save_tex(ss,'decay_curves')
	def generate_flux_detailed(self):
		ss = ''
		chnl = {'115INm':r'$^{115}$In(n,n'+"'"+r')$^{115m}$In','113INm':r'$^{113}$In(n,n'+"'"+r')$^{113m}$In','116INm':r'$^{115}$In(n,$\gamma$)$^{116m}$In','58CO':r'$^{58}$Ni(n,p)$^{58}$Co'}
		for smpl,fnm in enumerate([str(i[0]) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]):
			for istp in [str(i[3]).split(';') for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=? AND filename=?',(self.directory,fnm))][0]:
				ss += str(smpl+1)+' & '+chnl[istp]+' & '+str(round(self.flux_detailed[fnm][istp]['E'],2))+r' $\pm$ '+str(round(self.flux_detailed[fnm][istp]['dE'],3))+' & '+self.flux_TeX(self.flux_detailed[fnm][istp]['phi'],self.flux_detailed[fnm][istp]['unc_phi'])+r' \\'+'\n'
		self.save_tex(ss,'flux_detailed')
	def generate_flux_plots(self):
		ss = ''
		for n,sm in enumerate([self.parse_experiment_file(i) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]):
			ss += r'\begin{figure}[htb]'+'\n'
			ss += r'\includegraphics[width=9cm]{monitors/'+str(int(sm['dx']))+str(int(sm['dy']))+str(int(sm['dz']))+str(sm['elbow'])+'.pdf}\n'
			ss += r'\caption{Energy spectrum, mean and $\pm 1\sigma_E$ for sample '+str(n+1)+' calculated by Monte Carlo method.}'+'\n'
			ss += r'\label{fig:Spectrum'+str(n+1)+'}\n'
			ss += r'\end{figure}'+'\n\n'
		self.save_tex(ss,'flux_plots')
	def generate_flux_summary(self):
		ss = ''
		for smpl,fnm in enumerate([[str(i[0]),float(i[11]),float(i[12])] for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]):
			istp = [str(i[3]).split(';') for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=? AND filename=?',(self.directory,fnm[0]))][0][0]
			ss += str(smpl+1)+' & '+str(round(self.flux_detailed[fnm[0]][istp]['E'],2))+r' $\pm$ '+str(round(self.flux_detailed[fnm[0]][istp]['dE'],3))+' & '+self.flux_TeX(fnm[1],fnm[2])+r' \\'+'\n'
		self.save_tex(ss,'flux_summary')
	def generate_graphics_path(self):
		ss = r'\graphicspath{{../../plots/'+self.directory+r'/}{../../plots/pictures/}}'
		self.save_tex(ss,'graphics_path')
	def generate_irradiation_history(self):
		ss = ''
		start = [[[int(t) for t in m.split(':')] for m in i[1].split('::')] for i in self.db.execute('SELECT * FROM irradiation_history WHERE directory=? ORDER BY stop_time ASC',(self.directory,))]
		start = [dtm.datetime(t[0][0],t[0][1],t[0][2],t[1][0],t[1][1],t[1][2]) for t in start]
		stop = [[[int(t) for t in m.split(':')] for m in i[2].split('::')] for i in self.db.execute('SELECT * FROM irradiation_history WHERE directory=? ORDER BY stop_time ASC',(self.directory,))]
		stop = [dtm.datetime(t[0][0],t[0][1],t[0][2],t[1][0],t[1][1],t[1][2]) for t in stop]
		N = len(start)
		for n in range(N):
			ss += str(start[n].strftime("%H:%M %m/%d/%Y"))+' & '+str(stop[n].strftime("%H:%M %m/%d/%Y"))+' & '+str(round((stop[n]-start[n]).total_seconds()/3600.0,1))+r' \\'+'\n'
		self.save_tex(ss,'irradiation_history')
	def generate_monitor_table(self):
		ss = ''
		chnl = {'115INm':r'$^{115}$In(n,n'+"'"+r')$^{115m}$In','113INm':r'$^{113}$In(n,n'+"'"+r')$^{113m}$In','116INm':r'$^{115}$In(n,$\gamma$)$^{116m}$In','58CO':r'$^{58}$Ni(n,p)$^{58}$Co'}
		for smpl,fnm in enumerate([str(i[0]) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]):
			for istp in [str(i[3]).split(';') for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=? AND filename=?',(self.directory,fnm))][0]:
				ss += str(smpl+1)+' & '+chnl[istp]+' & '+str(round(self.flux_detailed[fnm][istp]['E'],2))+r' $\pm$ '+str(round(self.flux_detailed[fnm][istp]['dE'],3))+' & '+str(round(self.flux_detailed[fnm][istp]['sig'],1))+r' $\pm$ '+str(round(self.flux_detailed[fnm][istp]['unc_sig'],1))+r' \\'+'\n'
		self.save_tex(ss,'monitor_table')
	def generate_monitor_xs(self):
		ss = ''
		chnl = {'115INm':r'$^{115}$In(n,n'+"'"+r')$^{115m}$In','113INm':r'$^{113}$In(n,n'+"'"+r')$^{113m}$In','116INm':r'$^{115}$In(n,$\gamma$)$^{116m}$In','58CO':r'$^{58}$Ni(n,p)$^{58}$Co'}
		chn = {'115INm':'115IN_n_inl_115INm','113INm':'113IN_n_inl_113INm','116INm':'115IN_n_g_116INm','58CO':'58NI_n_p_58CO'}
		for istp in [str(i[3]).split(';') for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))][0]:
			ss += r'\begin{figure}[htb]'+'\n'
			ss += r'\includegraphics[width=9cm]{cross_sections/'+chn[istp]+'.pdf}'+'\n'
			ss += r'\caption{'+chnl[istp]+' cross section.  Data points are from EXFOR, solid/dashed lines are interpolated and $\pm 1\sigma$ values.}'+'\n'
			ss += r'\label{fig:'+chn[istp]+'}'+'\n'
			ss += r'\end{figure}'+'\n\n'
		self.save_tex(ss,'monitor_xs')
	def generate_peak_fit_plots(self):
		ss = ''
		for fnm in [str(i[0]) for i in self.db.execute('SELECT * FROM calibration_files WHERE directory=?',(self.directory,))]:
			ss += r'\begin{figure}[htb]'+'\n'
			ss += r'\includegraphics[width=\textwidth]{peak_fits/'+fnm.split('.')[0]+'.pdf}\n'
			ss += r'\caption{$\gamma$-ray energy spectrum and peak fits in \texttt{\detokenize{'+fnm+r'}}}'+'\n'
			ss += r'\label{fig:'+fnm.split('.')[0]+'}\n'
			ss += r'\end{figure}'+'\n\n'
		for fnm in [str(i[0]) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]:
			ss += r'\begin{figure}[htb]'+'\n'
			ss += r'\includegraphics[width=\textwidth]{peak_fits/'+fnm.split('.')[0]+'.pdf}\n'
			ss += r'\caption{$\gamma$-ray energy spectrum and peak fits in \texttt{\detokenize{'+fnm+r'}}}'+'\n'
			ss += r'\label{fig:'+fnm.split('.')[0]+'}\n'
			ss += r'\end{figure}'+'\n\n'
		self.save_tex(ss,'peak_fit_plots')
	def generate_peak_table(self):
		ss = ''
		for pk in [self.parse_calibration_peaks(i) for i in self.db.execute('SELECT * FROM calibration_peaks WHERE directory=?',(self.directory,))]:
			ss += ' & '.join([r'\texttt{\detokenize{'+pk['fnm']+'}}',isotope(pk['istp']).TeX(),str(pk['eng']),str(pk['I'])+r'$\pm$'+str(pk['unc_I']),str(pk['N'])+r'$\pm$'+str(pk['unc_N']),str(round(pk['chi2'],1))])+r' \\'+'\n'
		for fl in [self.parse_experiment_file(i) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]:
			for pk in [self.parse_experiment_peaks(i) for i in self.db.execute('SELECT * FROM experiment_peaks WHERE filename=?',(fl['f'],))]:
				ss += ' & '.join([r'\texttt{\detokenize{'+pk['fnm']+'}}',isotope(pk['istp']).TeX(),str(pk['eng']),str(pk['I'])+r'$\pm$'+str(pk['unc_I']),str(int(pk['N']))+r'$\pm$'+str(int(pk['unc_N'])),str(round(pk['chi2'],1))])+r' \\'+'\n'
		self.save_tex(ss,'peak_table')
	def generate_sample_mass(self):
		ss = ''
		chnl = {'115INm':'In','113INm':'In','116INm':'In','58CO':'Ni'}
		for n,fl in enumerate([self.parse_experiment_file(i) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]):
			ss += str(n+1)+' & '+chnl[fl['i'][0]]+' & '+str(fl['m'])+r' & \texttt{\detokenize{'+fl['f']+r'}} \\'+'\n'
		self.save_tex(ss,'sample_mass')
	def generate_sample_positions(self):
		ss = ''
		for n,fl in enumerate([self.parse_experiment_file(i) for i in self.db.execute('SELECT * FROM experiment_files WHERE directory=?',(self.directory,))]):
			ss += str(n+1)+' & '+str(fl['R_smpl'])+' & '+str(fl['dx'])+' & '+str(fl['dy'])+' & '+str(fl['dz'])+' & '+('90' if fl['elbow'] else '0')+r' \\'+'\n'
		self.save_tex(ss,'sample_positions')
	def generate_tex_files(self):
		os.chdir('../reports/'+self.directory)
		self.generate_abstract()
		self.generate_batemann()
		self.generate_decay_curves()
		self.generate_flux_detailed()
		self.generate_flux_plots()
		self.generate_flux_summary()
		self.generate_graphics_path()
		self.generate_irradiation_history()
		self.generate_monitor_table()
		self.generate_monitor_xs()
		self.generate_peak_fit_plots()
		self.generate_peak_table()
		self.generate_sample_mass()
		self.generate_sample_positions()
		os.chdir('../../code/')
	def run_latex(self):
		os.chdir('../reports/'+self.directory)
		os.putenv('PATH',os.getenv('PATH')+':/usr/local/texlive/2017/bin/x86_64-linux')
		for i in range(3):
			os.system('pdflatex -synctex=1 -interaction=nonstopmode '+self.directory+'_report.tex')
		os.chdir('../../code/')
		os.system('cp ../reports/'+self.directory+'/'+self.directory+'_report.pdf ../reports/'+self.directory+'_report.pdf')
	def generate_report(self):
		self.update_files()
		self.create_plots_dir()
		self.update_calibration()
		self.fit_experiment_peaks()
		self.calculate_flux()
		self.generate_tex_files()
		self.run_latex()


# fc = flux_calc()
# print fc.get_average_xs('115IN_n_g_116INm')
# fc.plot_energy_angle(saveplot=False,show=True)
# fc.plot_intensity_angle(saveplot=False,show=True)
# fc.plot_energy_spectrum(saveplot=False,show=True)

# sp = spectrum('Eu_centered.Spe')
# sp.plot_fits(wfit=True)

expt = experiment('112017_Cantarini_indium')
# expt = experiment('031218_Daniel_Ni')
# expt.update_files()
# expt.create_plots_dir()
# expt.update_calibration()
# expt.fit_experiment_peaks()
# expt.calculate_flux()
expt.generate_report()

