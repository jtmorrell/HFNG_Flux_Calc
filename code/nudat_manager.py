import matplotlib.pyplot as plt, numpy as np, matplotlib
import re,urllib2
from scipy.interpolate import interp1d
from os import walk
import sqlite3
matplotlib.rc('font',**{'size':14,'family':'sans-serif'})
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'
plt.rcParams['legend.fontsize']='12'


class exfor_xs(object):
	def __init__(self):
		self.pallate = {'k':'#2c3e50','b':'#2980b9','r':'#c0392b','y':'#f39c12','p':'#8e44ad','g':'#27ae60','gy':'#7f8c8d','o':'#d35400','w':'#ecf0f1','aq':'#16a085'}
	def TeX(self,istp):
		self.tex_g,self.metastable = False,False
		if istp.endswith('m'):
			self.metastable = True
			self.istp = istp[:-1].upper()
		elif istp.endswith('g'):
			self.istp = istp[:-1].upper()
			self.tex_g = True
		else:
			self.istp = istp.upper()
		el = ''.join(re.findall('[A-Z]+',self.istp))
		return r'$^{'+self.istp.split(el)[0]+('m' if self.metastable else '')+('g' if self.tex_g else '')+r'}$'+el.title()
	def parse_exfor(self,fnm):
		x4 = []
		for ln in open('../data/cross_sections/'+fnm,'r').read().split('\n'):
			if not ln.startswith('#') and not ln.startswith('//'):
				d = [i.strip() for i in ln.split(' ') if i.strip()!='']
				x4.append([float(d[0]),(float(d[1]) if float(d[1])>0 else 0.05*float(d[0])+0.01),1e3*float(d[2]),(1e3*float(d[3]) if float(d[3])>0 else 0.05*1e3*float(d[2])+0.01),d[5]])
		x4 = sorted(x4,key=lambda h:h[0])
		return {'E':[i[0] for i in x4],'dE':[i[1] for i in x4],'XS':[i[2] for i in x4],'dXS':[i[3] for i in x4],'auth':[i[4] for i in x4]}
	def exp_smooth(self,ls,alpha=0.3):
		R,RR,b = [ls[0]],[ls[-1]],1.0-alpha
		for i,ii in zip(ls[1:],reversed(ls[:-1])):
			R.append(alpha*i+b*R[-1])
			RR.append(alpha*ii+b*RR[-1])
		return [0.5*(R[n]+r) for n,r in enumerate(reversed(RR))]
	def get_smooth_xs(self,x4):
		mn,mx = x4['XS'][0],x4['XS'][-1]
		E_range = np.exp(np.arange(np.log(min(x4['E'])),np.log(max(x4['E']))+0.025,0.025))
		mu_sig = interp1d(x4['E'],[i**2 for i in self.exp_smooth(np.sqrt(x4['XS']),0.25)])
		for itr in range(3):
			R,E_R = [],[]
			for n,e in enumerate(E_range[:-1]):
				xs = [(x,x4['dXS'][m],x4['E'][m]) for m,x in enumerate(x4['XS']) if e<x4['E'][m]<E_range[n+1]]
				if len(xs)>0:
					E_R.append(0.5*(e+E_range[n+1]))
					R.append(np.average([i[0] for i in xs],weights=[1.0/(1e-9+(mu_sig(i[2])-i[0])**2) for i in xs]))
			mu_sig = interp1d(E_R,self.exp_smooth(R,0.65),kind='slinear',bounds_error=False,fill_value=(mn,mx))
		return interp1d(E_range,[i**2 for i in self.exp_smooth(np.sqrt(mu_sig(E_range)),0.65)],kind='quadratic',bounds_error=False,fill_value=(mn,mx))
	def get_error_xs(self,x4):
		sm_dXS = self.exp_smooth(x4['dXS'],0.1)
		return interp1d(x4['E'],sm_dXS,bounds_error=False,fill_value=(sm_dXS[0],sm_dXS[-1]))
	def plot_exfor(self,reaction='115IN_n_inl_115INm',show=False,save=False):
		x4 = self.parse_exfor(reaction+'.x4')
		f,ax = plt.subplots()
		ax.errorbar(x4['E'],x4['XS'],xerr=x4['dE'],yerr=x4['dXS'],ls='None',marker='.',color=self.pallate['gy'],label='EXFOR Data',zorder=1)
		mu_sig = self.get_smooth_xs(x4)
		unc_sig = self.get_error_xs(x4)
		E_range = np.arange(min(x4['E']),max(x4['E']),(max(x4['E'])-min(x4['E']))/float(len(x4['E'])))
		ax.plot(E_range,mu_sig(E_range),color=self.pallate['k'],lw=2.0,label='Interpolation',zorder=10)
		ax.plot(E_range,[mu_sig(e)+unc_sig(e) for e in E_range],ls='--',color=self.pallate['k'],lw=2.0,label=r'$\pm 1\sigma_{Interp}$',zorder=10)
		ax.plot(E_range,[mu_sig(e)-unc_sig(e) for e in E_range],ls='--',color=self.pallate['k'],lw=2.0,zorder=10)
		ax.set_xlabel('Neutron Energy [MeV]')
		ax.set_ylabel('Cross Section [mb]')
		rx = reaction.split('_')
		ax.set_title(self.TeX(rx[0])+'('+rx[1]+','+rx[2].replace("inl","n'").replace('g',r'$\gamma$')+')'+self.TeX(rx[3]))
		ax.legend(loc=0)
		f.tight_layout()
		if save:
			f.savefig('../plots/cross_sections/'+reaction+'.png')
			f.savefig('../plots/cross_sections/'+reaction+'.pdf')
		if show:
			plt.show()
		

class manager(object):
	def __init__(self):
		self.db_connection = sqlite3.connect('../data/nudat.db')
		self.db = self.db_connection.cursor()
	def update_xs_prediction(self,show=False):
		xs = exfor_xs()
		for reac in ['113IN_n_inl_113INm','58NI_n_p_58CO','115IN_n_inl_115INm','115IN_n_g_116INm']:
			xs.plot_exfor(reaction=reac,show=show)
			x4 = xs.parse_exfor(reac+'.x4')
			mn,mx = min(x4['E']),max(x4['E'])
			E_range = np.arange(mn,mx+0.05,0.05)
			avg,err = xs.get_smooth_xs(x4),xs.get_error_xs(x4)
			target,product = reac.split('_')[0],reac.split('_')[-1]
			self.db.executemany('INSERT INTO monitor_xs VALUES(?,?,?,?,?)',[(target,product,e,float(avg(e)),float(err(e))) for e in E_range])
		self.db_connection.commit()
		print 'Cross Section prediction up to date'
	def clear_second_isomers(self):
		itps = list(set([str(i[0]) for i in self.db.execute('SELECT * FROM ensdf')]))
		for itp in itps:
			E_parent = sorted(list(set([float(i[1]) for i in self.db.execute('SELECT * FROM ensdf WHERE isotope=?',(itp,))])))
			if len(E_parent)>2:
				for E in E_parent[2:]:
					self.db.execute('DELETE FROM ensdf WHERE isotope=? AND E_parent=?',(itp,E))
				self.db_connection.commit()
	def update_ensdf(self,m2=False):
		for nucl in ['133BA','152EU','60CO','137CS','54MN','241AM','115IN','113IN','116IN','58CO']:
			print 'Updating ',nucl
			self.search_ensdf(nucl)
		if not m2:
			self.clear_second_isomers()
		print 'ENSDF decay radiation up to date'
	def ensdf_multiplier(self,s):
		if 'E' in s:
			if '.' in s:
				return 10**(float(s.split('E')[1])-len(s.split('E')[0].split('.')[1]))
			return 10**(float(s.split('E')[1]))
		elif '.' in s:
			return 10**(-1*len(s.split('.')[1]))
		return 1.0
	def search_ensdf(self,nucl='133BA'):
		link = 'http://www.nndc.bnl.gov/chart/decaysearchdirect.jsp?nuc={0}&unc=nds'.format(nucl)
		response = urllib2.urlopen(link)
		self.db.execute('DELETE FROM ensdf WHERE isotope=?',(nucl,))
		for datum in response.read().split('Dataset')[1:]:
			data = datum.split('\n')
			eng,t_half = (i.split('white>')[1].split('<i>')[0].replace('&nbsp;','').strip() for n,i in enumerate(data[5].split('<td')) if n==4 or n==6)
			unc_t_half = [i.split('white>')[1].split('<i>')[1].split('</i>')[0] for n,i in enumerate(data[5].split('<td')) if n==6][0]
			unc_t_half = float(unc_t_half)*self.ensdf_multiplier(t_half.split(' ')[0].replace('+',''))*{'ms':0.001,'s':1.0,'m':60.0,'h':3600.0,'d':86400.0,'y':31557600.0}[t_half.split(' ')[1]]
			eng,t_half = float(eng),round(float(t_half.split(' ')[0])*{'ms':0.001,'s':1.0,'m':60.0,'h':3600.0,'d':86400.0,'y':31557600.0}[t_half.split(' ')[1]],2)
			gammas = []
			start,N = False,0
			for ln in data:
				if ln=='<p><u>Gamma and X-ray radiation</u>:':
					start = True
				elif ln=='</table>' and start:
					start = False
					break
				if start:
					if ln.startswith('<td nowrap'):
						if N==0:
							N+=1
						elif N==1:
							gammas.append([float(ln.split('align>')[1].split(' ')[0].replace('&nbsp;',''))])
							N+=1
						elif N==2:
							gammas[-1].append(float(ln.split('align>')[1].split(' ')[0].replace('&nbsp;','')))
							if ln.split('%')[1]!='</td>':
								gammas[-1].append(float(ln.split('<i>')[1].split('</i>')[0])*self.ensdf_multiplier(ln.split('align>')[1].split(' ')[0].replace('&nbsp;','')))
							else:
								gammas[-1].append(0.1*gammas[-1][-1])
							N+=1
						else:
							N=0
			self.db.executemany('INSERT INTO ensdf VALUES(?,?,?,?,?,?,?)',[(nucl,eng,t_half,g[0],g[1],g[2],unc_t_half) for g in gammas])
		self.db_connection.commit()

if __name__ == "__main__":
	mn = manager()
	# mn.update_ensdf()
	mn.update_xs_prediction(show=True)