#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <sys/errno.h>
#include <sys/wait.h>
#include "check_syscalls.h"
#include "particle.h"
#include "rockstar.h"
#include "groupies.h"
#include "inet/rsocket.h"
#include "inet/socket.h"
#include "io/meta_io.h"
#include "io/io_bgc2.h"
#include "bitarray.h"
#include "config_vars.h"
#include "server.h"
#include "client.h"
#include "merger.h"
#include "distance.h"
#include "bounds.h"
#include "fun_times.h"
#include "universe_time.h"
#include "interleaving.h"
#include "config.h"

#define CLIENT_DEBUG 0

extern struct rsocket *rsockets;

struct recipient *recipients = NULL;
int64_t num_recipients = 0;
int64_t *part_ids = NULL;
int64_t *part_id_buffer = NULL;
int64_t num_buffer_part_ids = 0;

struct chunk_info *chunk_info = NULL;
struct projection *prj = NULL;
struct projection_request *prq = NULL;
int64_t num_proj = 0;
int64_t in_error_state = 0;
int64_t RECIPIENT_BUFFER=100000;

FILE *profile_out = NULL;

void network_io_err(int64_t s) {
  if (in_error_state) return;
  in_error_state = 1;
  send_to_socket(s, "err!", 4);
}

void network_error_cleanup() {
  p = check_realloc(p, 0, "Freeing particle memory");
  num_p = 0;
  halos = check_realloc(halos, 0, "Freeing halos");
  num_halos = 0;
  rockstar_cleanup();
  clear_merger_tree();
  in_error_state = 0;
}

void reset_projection_count(void) {
  prj = check_realloc(prj, sizeof(struct projection)*num_proj,
		      "Allocating projections.");
  prq = check_realloc(prq, sizeof(struct projection_request)*num_proj,
		      "Allocating projection requests.");
}

struct recipient *add_recipient(int64_t struct_size, int64_t c) {
  struct recipient *r;
  num_recipients++;
  recipients = check_realloc(recipients, (sizeof(struct recipient)
					  * num_recipients), 
			     "Allocating particle data recipients.");
  r = recipients+num_recipients-1;
  memset(r, 0, sizeof(struct recipient));
  r->buffer = check_realloc(NULL, struct_size*RECIPIENT_BUFFER,
			    "Allocating recipient transmit buffer.");

  recv_from_socket(c, r->bounds, sizeof(float)*6);
  recv_from_socket(c, &(r->chunk), sizeof(int64_t));
  r->address = recv_msg_nolength(c, r->address);
  r->port = recv_msg_nolength(c, r->port);
  return r;
}

void clear_recipients(void) {
  int64_t i;
  for (i=0; i<num_recipients; i++) {
    free(recipients[i].buffer);
    free(recipients[i].port);
    free(recipients[i].address);
  }
  free(recipients);
  recipients = NULL;
  num_recipients = 0;
}

void calc_particle_bounds(float *bounds) {
  int64_t i,j;
  for (j=0; j<6; j++) bounds[j]=0;
  if (!num_p) return;
  memcpy(bounds, p[0].pos, sizeof(float)*3);
  memcpy(bounds+3, p[0].pos, sizeof(float)*3);
  for (i=1; i<num_p; i++) {
    for (j=0; j<3; j++) {
      if (bounds[j] > p[i].pos[j]) 
	bounds[j] = p[i].pos[j];
      if (bounds[j+3] < p[i].pos[j]) 
	bounds[j+3] = p[i].pos[j];
    }
  }
}

void calc_particle_bounds_periodic(float *bounds) {
  int64_t i,j;
  for (j=0; j<6; j++) bounds[j]=0;
  if (!num_p) return;
  memcpy(bounds, p[0].pos, sizeof(float)*3);
  memcpy(bounds+3, p[0].pos, sizeof(float)*3);
  for (i=1; i<num_p; i++) {
    for (j=0; j<3; j++) {
      float pos = p[i].pos[j];
      if (p[0].pos[j]-pos > BOX_SIZE/2.0) pos+=BOX_SIZE;
      else if (p[0].pos[j]-pos < -BOX_SIZE/2.0) pos-=BOX_SIZE;
      if (bounds[j] > pos) bounds[j] = pos;
      if (bounds[j+3] < pos) bounds[j+3] = pos;
    }
  }
}

void calc_halo_bounds(float *bounds) {
  int64_t i,j;
  for (j=0; j<6; j++) bounds[j]=0;
  if (!num_halos) return;
  memcpy(bounds, halos[0].pos, sizeof(float)*3);
  memcpy(bounds+3, halos[0].pos, sizeof(float)*3);
  for (i=0; i<num_halos; i++) {
    float r = BGC2_R*halos[i].r;
    if (STRICT_SO_MASSES) r = BGC2_R*max_halo_radius(halos+i);
    for (j=0; j<3; j++) {
      if (bounds[j] > halos[i].pos[j]-r) bounds[j] = halos[i].pos[j]-r;
      if (bounds[j+3] < halos[i].pos[j]+r) bounds[j+3] = halos[i].pos[j]+r;
    }
  }
}

void trim_particles(float *bounds) {
  int64_t i;
  if (!TRIM_OVERLAP) return;
  for (i=0; i<3; i++) {
    bounds[i] += TRIM_OVERLAP;
    bounds[i+3] -= TRIM_OVERLAP;
  }

  if (ROUND_AFTER_TRIM)
    for (i=0; i<6; i++) 
      bounds[i]=((int64_t)(bounds[i]/ROUND_AFTER_TRIM + 0.5))*ROUND_AFTER_TRIM;

  for (i=0; i<num_p; i++)
    if (!_check_bounds_raw(p[i].pos, bounds)) {
	num_p--;
	p[i] = p[num_p];
	i--;
    }

  p = check_realloc(p, sizeof(struct particle)*num_p, "Removing overlap.");
}

void clear_particle_rbuffer(struct recipient *r) {
  if (!r->buffered) return;
  send_to_socket_noconfirm(r->cs, "part", 4);
  send_to_socket_noconfirm(r->cs, r->buffer, sizeof(struct particle)*r->buffered);
  r->buffered = 0;
}

void clear_bparticle_rbuffer(struct recipient *r) {
  if (!r->buffered) return;
  send_to_socket_noconfirm(r->cs, "bprt", 4);
  send_to_socket_noconfirm(r->cs, r->buffer, sizeof(struct bparticle)*r->buffered);
  r->buffered = 0;
}

void clear_halo_rbuffer(struct recipient *r) {
  int64_t i, pids=0;
  struct halo *bh = r->buffer;
  if (!r->buffered) return;
  send_to_socket_noconfirm(r->cs, "halo", 4);
  send_to_socket_noconfirm(r->cs, r->buffer, sizeof(struct halo)*r->buffered);

  for (i=0; i<r->buffered; i++) pids+=bh[i].num_p;
  if (pids>num_buffer_part_ids) {
    part_id_buffer = check_realloc(part_id_buffer, sizeof(int64_t)*pids,
				   "Allocating particle ID buffer.");
    num_buffer_part_ids = pids;
  }
  pids = 0;
  for (i=0; i<r->buffered; i++) {
    memcpy(part_id_buffer + pids, part_ids + bh[i].p_start,
	   sizeof(int64_t)*bh[i].num_p);
    pids += bh[i].num_p;
  }
  send_to_socket_noconfirm(r->cs, "pids", 4);
  send_to_socket_noconfirm(r->cs, part_id_buffer, sizeof(int64_t)*pids);
  r->buffered = 0;
}


void add_sp_to_buffer(struct recipient *r, struct sphere_request *sp) {
  assert(r->buffered >= 0);
  if (!(r->buffered%1000)) 
    r->buffer = check_realloc(r->buffer, sizeof(struct sphere_request)*
			      (r->buffered+1000), "Sphere request buffer");
  struct sphere_request *buffer = r->buffer;
  buffer[r->buffered] = *sp;
  r->buffered++;
}

void add_particle_to_buffer(struct recipient *r, struct particle *p1) {
  struct particle *buffer = r->buffer;
  if (r->buffered == RECIPIENT_BUFFER) clear_particle_rbuffer(r); 
  buffer[r->buffered] = *p1;
  r->buffered++;
}

void add_halo_to_buffer(struct recipient *r, struct halo *h1) {
  struct halo *buffer = r->buffer;
  if (r->buffered == RECIPIENT_BUFFER) clear_halo_rbuffer(r); 
  buffer[r->buffered] = *h1;
  r->buffered++;
}

void add_bparticle_to_buffer(struct recipient *r, struct bparticle *tbp) {
  struct bparticle *buffer = r->buffer;
  if (r->buffered == RECIPIENT_BUFFER) clear_bparticle_rbuffer(r); 
  buffer[r->buffered] = *tbp;
  r->buffered++;
}

int64_t check_particle_bounds(struct particle *p1, struct recipient *r) {
  if (_check_bounds_raw(p1->pos, r->bounds)) {
    add_particle_to_buffer(r, p1);
    return 1;
  }
  return 0;
}

//No longer corrects periodic!!!!
void check_bparticle_bounds(struct bparticle *tbp, struct recipient *r) {
  struct bparticle pt = *tbp;
  //  if (_check_bounds(tbp->pos, pt.pos, r->bounds)) add_bparticle_to_buffer(r, &pt);
  if (_check_bounds(tbp->pos, pt.pos, r->bounds)) add_bparticle_to_buffer(r, tbp);
}

void check_halo_bounds(struct halo *h1, struct recipient *r) {
  struct halo ht = *h1;
  if (_check_bounds(h1->pos, ht.pos, r->bounds)) add_halo_to_buffer(r, &ht);
}

void check_bgc2_bounds(struct halo *h1, struct recipient *r) {
  struct sphere_request sp;
  float bounds[6];
  int64_t i;
  sp.r = h1->r * BGC2_R;
  if (STRICT_SO_MASSES) sp.r = BGC2_R * max_halo_radius(h1);
  for (i=0; i<3; i++) {
    bounds[i] = r->bounds[i]-sp.r;
    bounds[i+3] = r->bounds[i+3]+sp.r;
  }
  if (_check_bounds(h1->pos, sp.cen, bounds)) add_sp_to_buffer(r, &sp);
}

int64_t check_projection_bounds(struct particle *p1, struct projection *pr) {
  return (_check_bounds_raw(p1->pos, pr->bounds));
}

void send_config(int64_t c) {
  double data;
#define snd(x) { data = x; send_to_socket_noconfirm(c, &data, sizeof(double)); }
  snd(PARTICLE_MASS);
  snd(AVG_PARTICLE_SPACING);
  snd(SCALE_NOW);
  snd(BOX_SIZE);
  snd(Ol);
  snd(Om);
  snd(h0);
  snd(TRIM_OVERLAP);
  snd(ROUND_AFTER_TRIM);
#undef snd
}

void recv_config(int64_t c) {
  int64_t i;
#define rcv(x) { recv_from_socket(c, &x, sizeof(double)); }
  rcv(PARTICLE_MASS);
  rcv(AVG_PARTICLE_SPACING);
  rcv(SCALE_NOW);
  rcv(BOX_SIZE);
  rcv(Ol);
  rcv(Om);
  rcv(h0);
  rcv(TRIM_OVERLAP);
  rcv(ROUND_AFTER_TRIM);
#undef rcv
 if (strlen(LIGHTCONE_ALT_SNAPS)) {
   for (i=0; i<3; i++)
     if (LIGHTCONE_ORIGIN[i] || LIGHTCONE_ALT_ORIGIN[i]) break;
   if (i==3) {
     for (i=0; i<3; i++) LIGHTCONE_ORIGIN[i] = LIGHTCONE_ALT_ORIGIN[i] = BOX_SIZE/2.0;
   }
 }
}

void send_particles(int64_t c, float *bounds) {
  int64_t i,j;
  for (j=0; j<num_recipients; j++)
    recipients[j].cs = 
      connect_to_addr(recipients[j].address, recipients[j].port);

  for (i=num_p-1; i>=0; i--) {
    for (j=0; j<num_recipients; j++)
      if (check_particle_bounds(p+i, recipients+j)) break;
    if (!(i%PARTICLE_REALLOC_NUM)) {
      p = check_realloc(p,sizeof(struct particle)*i,"Freeing particle memory.");
      check_mtrim();
    }
  }
  num_p = 0;
  p = check_realloc(p,0,"Freeing particle memory.");
  check_mtrim();
  for (j=0; j<num_recipients; j++) {
    clear_particle_rbuffer(recipients+j);
    send_to_socket(recipients[j].cs, "done", 4);
    close_rsocket(recipients[j].cs);
  }
  clear_recipients();
  send_to_socket(c, "done", 4);
}

void send_bparticles(char *c_address, char *c_port) {
  int64_t i,j, c;

  for (j=0; j<num_recipients; j++) {
    recipients[j].cs=connect_to_addr(recipients[j].address, recipients[j].port);
  }

  for (i=0; i<num_bp; i++)
    for (j=0; j<num_recipients; j++) check_bparticle_bounds(bp+i, recipients+j);

  for (j=0; j<num_recipients; j++) {
    clear_bparticle_rbuffer(recipients+j);
    send_to_socket(recipients[j].cs, "done", 4);
    close_rsocket(recipients[j].cs);
  }
  
  c = connect_to_addr(c_address, c_port);
  send_to_socket(c, "rdne", 4);
  exit(0);
}

void gather_spheres(char *c_address, char *c_port, float *bounds, int64_t id_offset, int64_t snap, int64_t chunk) {
  int64_t i,j,k,c;

  for (i=0; i<num_halos; i++) {
    if (!_should_print(halos+i, bounds)) continue;
    float r = BGC2_R*halos[i].r;
    if (STRICT_SO_MASSES) r = BGC2_R*max_halo_radius(halos+i);
    for (j=0; j<3; j++) {
      if (halos[i].pos[j]-r < bounds[j]) break;
      if (halos[i].pos[j]+r > bounds[j+3]) break;
    }
    if (j==3) continue;
    for (j=0; j<num_recipients; j++) {
      if (recipients[j].chunk == our_chunk) continue;
      check_bgc2_bounds(halos+i, recipients+j);
    }
  }

  for (j=0; j<num_recipients; j++) {
    if (recipients[j].chunk == our_chunk) continue;
    c = connect_to_addr(recipients[j].address, recipients[j].port);
    send_to_socket_noconfirm(c, "sphr", 4);
    send_to_socket_noconfirm(c, recipients[j].buffer,
	       sizeof(struct sphere_request)*recipients[j].buffered);
    k=1;
    while (1) {
      recv_from_socket(c, &k, sizeof(int64_t));
      if (!k) break;
      ep2 = check_realloc(ep2, sizeof(struct extended_particle)*(num_ep2+k),
			  "Allocating secondary extended particles.");
      recv_from_socket(c, ep2+num_ep2, sizeof(struct extended_particle)*k);
      num_ep2+=k;
    }
    send_to_socket_noconfirm(c, "done", 4);
    close_rsocket(c);
  }

  output_bgc2(id_offset, snap, chunk, bounds);
  
  c = connect_to_addr(c_address, c_port);
  //  send_to_socket_noconfirm(c, "mass", 4);
  //send_to_socket(c, halos, sizeof(struct halo)*num_halos);
  send_to_socket(c, "rdne", 4);
  exit(0);
}


int sort_by_chunk(const void *a, const void *b) {
  const struct bgroup *c = a;
  const struct bgroup *d = b;
  if (c->chunk < d->chunk) return -1;
  if (c->chunk > d->chunk) return 1;
  if (c->id < d->id) return -1;
  if (c->id > d->id) return 1;
  //assert(c->id != d->id);
  return 0;
}


void collect_bgroups(int64_t chunk) {
  int64_t i,j,k, c, connect_chunk=chunk, total_bg;
  free(p);
  rockstar_cleanup();
  bgroups_to_setlist();
  clear_bg_data();
  for (i=0,total_bg=0; i<num_bg_sets; i++) total_bg+=bg_set_sizes[i];
  while (num_bg_sets && total_bg) {
    c = connect_to_addr(chunk_info[connect_chunk].address,
			chunk_info[connect_chunk].port);
    send_to_socket_noconfirm(c, "lnkb", 4);
    send_to_socket_noconfirm(c, &chunk, sizeof(int64_t));
    send_to_socket_noconfirm(c, &num_bg_sets, sizeof(int64_t));
    send_to_socket_noconfirm(c, bg_set_sizes, sizeof(int64_t)*num_bg_sets);
    for (i=0,total_bg=0; i<num_bg_sets; i++) total_bg+=bg_set_sizes[i];
    send_to_socket_noconfirm(c, final_bg, sizeof(struct bgroup)*total_bg);
    recv_from_socket(c, &num_bg_sets, sizeof(int64_t));
    if (num_bg_sets) {
      bg_set_sizes = recv_and_alloc(c, bg_set_sizes, sizeof(int64_t)*num_bg_sets);
      final_bg = recv_msg_nolength(c, final_bg);
    }
    send_to_socket(c, "done", 4);
    close_rsocket(c);
    connect_chunk = calc_next_bgroup_chunk();
    if (connect_chunk < 0) break;
    for (i=0,total_bg=0; i<num_bg_sets; i++) total_bg+=bg_set_sizes[i];
  }

  num_bg_sets = prune_setlist();
  int64_t *set_chunks = NULL, *sets_per_chunk = NULL, *chunk_indices = NULL;
  check_realloc_s(set_chunks, sizeof(int64_t), num_bg_sets);
  check_realloc_s(sets_per_chunk, sizeof(int64_t), NUM_WRITERS);
  check_realloc_s(chunk_indices, sizeof(int64_t), NUM_WRITERS);
  for (i=0; i<NUM_WRITERS; i++) sets_per_chunk[i] = chunk_indices[i] = 0;
  
  //Sort bg sets by the chunk with max # of particles
  j=0;
  for (i=0; i<num_bg_sets; i++) {
    int64_t max_c = final_bg[j].chunk;
    int64_t max_p = final_bg[j].num_p;
    int64_t cur_c = max_c, cur_p = 0;

    assert(bg_set_sizes[i]);
    qsort(final_bg+j, bg_set_sizes[i], sizeof(struct bgroup), sort_by_chunk);
    assert(final_bg[j].chunk == chunk);

    cur_c = final_bg[j].chunk;
    for (k=j; k<j+bg_set_sizes[i]; k++) {
      if (cur_c != final_bg[k].chunk) {
	if (cur_p > max_p) {
	  max_c = cur_c;
	  max_p = cur_p;
	}
	cur_c = final_bg[k].chunk;
	cur_p = 0;
      }
      cur_p += final_bg[k].num_p;
    }
    if (cur_p > max_p) max_c = cur_c;
    set_chunks[i] = max_c;
    sets_per_chunk[max_c]+=bg_set_sizes[i];
    j+=bg_set_sizes[i];
  }

  for (i=1; i<NUM_WRITERS; i++)
    chunk_indices[i] = chunk_indices[i-1]+sets_per_chunk[i-1];
  for (i=0,total_bg=0; i<num_bg_sets; i++) total_bg+=bg_set_sizes[i];

  //Reorder sets
  struct bgroup *new_groups = NULL;
  check_realloc_s(new_groups, sizeof(struct bgroup), total_bg);
  j=0;
  for (i=0; i<num_bg_sets; i++) {
    for (k=j; k<j+bg_set_sizes[i]; k++) {
      new_groups[chunk_indices[set_chunks[i]]] = final_bg[k];
      chunk_indices[set_chunks[i]]++;
    }
    j+=bg_set_sizes[i];
  }
  free(final_bg);
  final_bg = new_groups;

  int64_t *new_set_sizes = NULL;
  check_realloc_s(new_set_sizes, sizeof(int64_t), num_bg_sets);
  for (i=0; i<NUM_WRITERS; i++) sets_per_chunk[i] = chunk_indices[i] = 0;
  for (i=0; i<num_bg_sets; i++) sets_per_chunk[set_chunks[i]]++;
  for (i=1; i<NUM_WRITERS; i++) chunk_indices[i] = chunk_indices[i-1]+sets_per_chunk[i-1];
  for (i=0; i<num_bg_sets; i++) {
    new_set_sizes[chunk_indices[set_chunks[i]]] = bg_set_sizes[i];
    chunk_indices[set_chunks[i]]++;
  }
  free(bg_set_sizes);
  free(set_chunks);
  bg_set_sizes = new_set_sizes;

  //Send sets to appropriate chunk
  int64_t chunk_offset = 0;
  for (connect_chunk=0; connect_chunk<NUM_WRITERS; connect_chunk++) {
    if (!sets_per_chunk[connect_chunk]) continue;
    c = connect_to_addr(chunk_info[connect_chunk].address,
		      chunk_info[connect_chunk].port);
    send_to_socket_noconfirm(c, "strb", 4);
    send_to_socket_noconfirm(c, &chunk, sizeof(int64_t));
    send_to_socket_noconfirm(c, sets_per_chunk+connect_chunk, sizeof(int64_t));
    chunk_indices[connect_chunk]-=sets_per_chunk[connect_chunk];
    send_to_socket_noconfirm(c, bg_set_sizes+chunk_indices[connect_chunk],
			     sizeof(int64_t)*sets_per_chunk[connect_chunk]);
    for (j=0,total_bg=0; j<sets_per_chunk[connect_chunk]; j++)
      total_bg+=bg_set_sizes[chunk_indices[connect_chunk]+j];
    send_to_socket_noconfirm(c, final_bg+chunk_offset, total_bg*sizeof(struct bgroup));
    chunk_offset += total_bg;
    send_to_socket(c, "done", 4);
    close_rsocket(c);
  }

  c = connect_to_addr(chunk_info[chunk].address,
		      chunk_info[chunk].port);
  send_to_socket(c, "rdne", 4);
  close_rsocket(c);
  exit(0);
}


void send_halos(char *c_address, char *c_port, int64_t snap, int64_t chunk) {
  int64_t i,j, c;
  struct binary_output_header bheader;

  load_binary_halos(snap, chunk, &bheader, &halos, &part_ids, 0);

  for (j=0; j<num_recipients; j++)
    recipients[j].cs=connect_to_addr(recipients[j].address, recipients[j].port);

  for (i=0; i<bheader.num_halos; i++)
    for (j=0; j<num_recipients; j++) check_halo_bounds(halos+i, recipients+j);

  for (j=0; j<num_recipients; j++) {
    clear_halo_rbuffer(recipients+j);
    send_to_socket_noconfirm(recipients[j].cs, "cnfg", 4);
    send_config(recipients[j].cs);
    send_to_socket(recipients[j].cs, "done", 4);
    close_rsocket(recipients[j].cs);
  }
  
  c = connect_to_addr(c_address, c_port);
  send_to_socket(c, "rdne", 4);
  exit(0);
}

void close_connection(int64_t cs, int64_t *cslist, int64_t *num_cs) {
  int64_t i;
  for (i=0; i<*num_cs; i++) {
    if (cslist[i] == cs) {
      close_rsocket(cslist[i]);
      *num_cs = (*num_cs)-1;
      cslist[i] = cslist[*num_cs];
      break;
    }
  }
}

void transfer_stuff(int64_t s, int64_t c, int64_t timestep) {
  int64_t i, j, k, num_senders = 0, done = 0, length, new_p_start, *senders = NULL;
  char cmd[5] = {0};
  struct binary_output_header *bheader;
  struct halo **halos_recv, *th;
  struct sphere_request *sp;
  struct extended_particle *epbuffer = NULL;
  int64_t max_conn, **pids_recv;
  char *bitarray = NULL;

  while (!in_error_state && (num_senders || !done)) {
    clear_rsocket_tags();
    if (!done) {
      tag_rsocket(s);
      tag_rsocket(c);
    }
    for (i=0; i<num_senders; i++) tag_rsocket(senders[i]);
    max_conn = select_rsocket(RSOCKET_READ, 0);
    for (i=0; i<max_conn; i++) {
      if (!check_rsocket_tag(i)) continue;
      if (i==s) {
	num_senders++;
	senders = check_realloc(senders, sizeof(int64_t)*num_senders,
				"Allocating particle sender FDs.");
	senders[num_senders-1] = accept_connection(s,NULL,NULL);
      }
      else if (i==c) {
	recv_from_socket(c, cmd, 4);
	if (!strcmp(cmd, "done")) { done = 1; }
	else if (!strcmp(cmd, "err!")) {
	  in_error_state = 1;
	  for (j=0; j<num_senders; j++) close_rsocket(senders[j]);
	  num_senders = 0;
	  break;
	}
	else { fprintf(stderr, "[Error] Server protocol error rs (%s)!\n", cmd); exit(1); }
      }
      else {
	if (recv_from_socket(i, cmd, 4)<=0) {
	  network_io_err(c);
	  for (j=0; j<num_senders; j++) close_rsocket(senders[j]);
	  num_senders = 0;
	  break;
	}
	if (!strcmp(cmd, "part")) {
	  length = num_p*sizeof(struct particle);
	  p = recv_msg(i, p, &length, length);
	  assert(!(length%(sizeof(struct particle))));
	  num_p = length / sizeof(struct particle);
	}

	else if (!strcmp(cmd, "sphr")) {
	  length = 0;
	  sp = recv_msg(i, NULL, &length, length);
	  assert(!(length%(sizeof(struct sphere_request))));
	  length /= sizeof(struct sphere_request);
	  if (!epbuffer) epbuffer = check_realloc(NULL, sizeof(struct extended_particle)*PARTICLE_REALLOC_NUM, "Particle buffer");
	  if (!bitarray) bitarray = BIT_ALLOC(num_p+num_additional_p);
	  BIT_ALL_CLEAR(bitarray, num_p+num_additional_p);
	  k=0;
	  for (j=0; j<length; j++) {
	    int64_t l, num_sp;
	    struct extended_particle **result = 
	      do_sphere_request(sp[j].cen, sp[j].r, &num_sp);
	    for (l=0; l<num_sp; l++) {
	      if (BIT_TST(bitarray, result[l]-ep)) continue;
	      BIT_SET(bitarray, result[l]-ep);
	      epbuffer[k] = result[l][0];
	      epbuffer[k].hid = -1;
	      k++;
	      if (k==PARTICLE_REALLOC_NUM) {
		send_to_socket_noconfirm(i, &k, sizeof(int64_t));
		send_to_socket_noconfirm(i, epbuffer, sizeof(struct extended_particle)*k);
		k=0;
	      }
	    }
	  }
	  if (k) {
	    send_to_socket_noconfirm(i, &k, sizeof(int64_t));
	    send_to_socket_noconfirm(i, epbuffer, sizeof(struct extended_particle)*k);
	    k=0;
	  }
	  send_to_socket_noconfirm(i, &k, sizeof(int64_t));
	  free(sp);
	}

	else if (!strcmp(cmd, "bprt")) {
	  length = num_bp*sizeof(struct bparticle);
	  bp = recv_msg(i, bp, &length, length);
	  assert(!(length%(sizeof(struct bparticle))));
	  int64_t old_bp = num_bp;
	  num_bp = length / sizeof(struct bparticle);
	  num_new_bp += num_bp - old_bp;
	}

	else if (!strcmp(cmd, "halo")) {
	  assert(timestep > 0);
	  bheader = (timestep > 1) ? &head2 : &head1;
	  pids_recv = (timestep > 1) ? &part2 : &part1;
	  halos_recv = (timestep > 1) ? &halos2 : &halos1;

	  length = bheader->num_halos*sizeof(struct halo);
	  *halos_recv = recv_msg(i, *halos_recv, &length, length);
	  assert(!(length%sizeof(struct halo)));
	  length /= sizeof(struct halo);

	  //Redo particle pointers
	  new_p_start = bheader->num_particles;
	  for (j=bheader->num_halos; j<length; j++) {
	    th = (*halos_recv) + j;
	    th->p_start = new_p_start;
	    new_p_start += th->num_p;
	  }
	  bheader->num_halos = length;

	  recv_from_socket(i, cmd, 4);
	  assert(!strcmp(cmd, "pids"));
	  length = bheader->num_particles*sizeof(int64_t);
	  *pids_recv = recv_msg(i, *pids_recv, &length, length);
	  assert(!(length%sizeof(int64_t)));
	  bheader->num_particles = length / sizeof(int64_t);
	}

	else if (!strcmp(cmd, "lnkb") || !strcmp(cmd, "strb")) {
	  int64_t num_sets = 0, client_chunk, total_groups, total_bg;
	  recv_from_socket(i, &client_chunk, sizeof(int64_t));
	  recv_from_socket(i, &num_sets, sizeof(int64_t));
	  int64_t *set_sizes = NULL;
	  struct bgroup *bgroup_request = NULL;
	  if (!strcmp(cmd, "lnkb")) {
	    if (num_sets) {
	      set_sizes = recv_and_alloc(i, NULL, sizeof(int64_t)*num_sets);
	      for (j=0,total_bg=0; j<num_sets; j++) total_bg += set_sizes[j];
	      bgroup_request = recv_and_alloc(i, NULL, sizeof(struct bgroup)*total_bg);
	    }
	    find_bgroup_sets(client_chunk, &num_sets, &set_sizes,
			     &bgroup_request, &total_groups);
	    send_to_socket_noconfirm(i, &num_sets, sizeof(int64_t));
	    if (num_sets) {
	      send_to_socket_noconfirm(i, set_sizes, sizeof(int64_t)*num_sets);
	      send_to_socket_noconfirm(i, bgroup_request,
			     sizeof(struct bgroup)*total_groups);
	      free(set_sizes);
	      free(bgroup_request);
	    }
	  }
	  else if (num_sets) {
	      check_realloc_s(bg_set_sizes, sizeof(int64_t), num_bg_sets+num_sets);
	      recv_from_socket(i, bg_set_sizes+num_bg_sets, sizeof(int64_t)*num_sets);
	      for (j=0,total_bg=0; j<num_bg_sets+num_sets; j++) total_bg += bg_set_sizes[j];
	      check_realloc_s(final_bg, sizeof(struct bgroup), total_bg);
	      bgroup_request = final_bg + total_bg;
	      for (j=0,total_bg=0; j<num_sets; j++) total_bg += bg_set_sizes[j+num_bg_sets];
	      bgroup_request -= total_bg;
	      recv_from_socket(i, bgroup_request, sizeof(struct bgroup)*total_bg);
	      num_bg_sets += num_sets;
	  }
	}

	else if (!strcmp(cmd, "cnfg")) {
	  recv_config(i);
	}

	else if (!strcmp(cmd, "done")) {
	  close_connection(i, senders, &num_senders);
	}

	else if (!strcmp(cmd, "rdne")) {
	  close_connection(i, senders, &num_senders);
	  send_to_socket_noconfirm(c, "done", 4);
	}

	else { fprintf(stderr, "[Error] Client protocol error rs (%s)!\n", cmd); exit(1); }
      }
    }
  }
  free(senders);
  if (bitarray) free(bitarray);
  if (epbuffer) free(epbuffer);
}

void do_projections(void) {
  int64_t i, j, idx, dir=0;
  assert(BOX_SIZE > 0);
  for (i=0; i<num_proj; i++) {
    prj[i].id = prq[i].id;
    dir = prj[i].dir = prq[i].dir;
    memcpy(prj[i].bounds, prq[i].bounds, sizeof(float)*6);
    for (j=0; j<PROJECTION_SIZE; j++) prj[i].data[j] = 0;
  }
  for (j=0; j<num_p; j++) {
    for (i=0; i<num_proj; i++) {
      if (check_projection_bounds(p+j, prj+i)) {
	idx = (double)PROJECTION_SIZE*p[j].pos[dir]/(double)BOX_SIZE;
	if (idx >= PROJECTION_SIZE) idx = PROJECTION_SIZE-1;
	prj[i].data[idx]++;
      }
    }
  }
}


void accept_workloads(char *c_address, char *c_port, int64_t snap, int64_t chunk) {
  char *address = NULL, *port = NULL;
  int64_t s = connect_to_addr(c_address, c_port);
  int64_t m = s;
  int64_t id = -1, new_bounds = 1;
  int64_t i, minus1 = -1, loc;
  char cmd[5] = {0};
  float zero_bounds[6];
  struct fof *fofs = NULL;
  struct workunit_info w;
  struct bgroup *bgroup_list = NULL;
  int64_t *chunks = NULL;
  int64_t num_chunks = 0;
  int64_t *set_sizes = NULL;
  struct particle *pbuffer = NULL;
  particle_cleanup();
  rockstar_cleanup();
  clear_final_bg_data();
  int64_t time_a, time_b;
  int64_t idle_time = 0, recv_time = 0, send_time = 0, bp_time = 0, f_time = 0,
    work_time = 0, ph_time = 0, total_pp = 0, total_h = 0, total_wku = 0;

  chunks = check_realloc(NULL, sizeof(int64_t)*NUM_WRITERS, "chunk ids");
  if (s < 0) exit(1);
  memset(zero_bounds, 0, sizeof(float)*6);

  time_a = time(NULL);
#define record_time(x) { time_b = time(NULL); (x) += time_b-time_a; time_a = time_b; }
  send_to_socket(m, &chunk, sizeof(int64_t));
  while (1) {
    if (recv_from_socket(m, cmd, 4)<=0) {
      fprintf(stderr, "[Warning] Failed to receive instruction from server for chunk %"PRId64" (shutting down)!\n", chunk);
      send_to_socket(s, "clos", 4);
      exit(1);
    }
    if (!strcmp(cmd, "wrku")) {
      record_time(idle_time);
      recv_from_socket(m, &w, sizeof(struct workunit_info));
      assert((w.num_particles+w.num_meta_p) >= 0 && w.num_fofs >= 0 && !w.num_halos);
      num_p = w.num_particles+w.num_meta_p;
      fofs = recv_and_alloc(m, fofs, sizeof(struct fof)*w.num_fofs);
      p = check_realloc(p, sizeof(struct particle)*num_p, "particles");
      assert(w.chunk >= 0 && w.chunk < NUM_WRITERS);
      memcpy(chunk_info[w.chunk].bounds, w.bounds, sizeof(float)*6);
      if (w.num_particles)
	recv_from_socket(m, p, sizeof(struct particle)*w.num_particles);
      record_time(recv_time);

      if (w.num_meta_fofs) {
	set_sizes=recv_and_alloc(m, set_sizes, sizeof(int64_t)*w.num_meta_fofs);
	bgroup_list = recv_and_alloc(m, bgroup_list, sizeof(struct bgroup)*w.total_bg);
	loc = 0;
	for (i=0; i<w.num_meta_fofs; i++) {
	  qsort(bgroup_list+loc, set_sizes[i], sizeof(struct bgroup), sort_by_chunk);
	  loc += set_sizes[i];
	}

	loc = w.num_particles;
	for (i=0; i<w.total_bg; i++) {
	  bgroup_list[i].tagged = loc;
	  loc += bgroup_list[i].num_p;
	}
	qsort(bgroup_list, w.total_bg, sizeof(struct bgroup), sort_by_chunk);
	int64_t dup_ids = 0;
	for (i=1; i<w.total_bg; i++) 
	  if (bgroup_list[i].chunk == bgroup_list[i-1].chunk &&
	      bgroup_list[i].id == bgroup_list[i-1].id) dup_ids=1;

	i=0;
	num_chunks = 0;
	while (i<w.total_bg) {
	  loc = i;
	  int64_t part_in_chunk = 0, part_received = 0;
	  for (; i<w.total_bg; i++) {
	    if (bgroup_list[loc].chunk != bgroup_list[i].chunk) break;
	    part_in_chunk += bgroup_list[i].num_p;
	  }
	  int64_t num_groups = i-loc;
	  if (!num_groups || !part_in_chunk) break;
	  int64_t cchunk = bgroup_list[loc].chunk;
	  chunks[num_chunks] = cchunk;
	  num_chunks++;
	  int64_t m2 = connect_to_addr(chunk_info[cchunk].address,
				       chunk_info[cchunk].port);
	  send_to_socket_noconfirm(m2, &minus1, sizeof(int64_t));
	  send_to_socket_noconfirm(m2, "part", 4);
	  send_to_socket_noconfirm(m2, &num_groups, sizeof(int64_t));
	  send_to_socket_noconfirm(m2, bgroup_list + loc, sizeof(struct bgroup)*num_groups);
	  recv_from_socket(m2, chunk_info[cchunk].bounds, sizeof(float)*6);

	  int64_t k=0,l=0;
	  while (part_received < part_in_chunk) {
	    int64_t this_received = 0;
	    pbuffer = recv_msg(m2, pbuffer, &this_received, 0);
	    assert((this_received % (sizeof(struct particle)))==0);
	    this_received /= sizeof(struct particle);
	    part_received += this_received;
	    for (l=0; l<this_received; l++,k++) {
	      while (k==bgroup_list[loc].num_p) { k=0; loc++; }
	      p[bgroup_list[loc].tagged+k] = pbuffer[l];
	    }
	  }
	  close_rsocket(m2);
	}
	if (memcmp(w.bounds, zero_bounds, sizeof(float)*6)!=0) {
	  if (!PERIODIC || !BOX_SIZE) calc_particle_bounds(w.bounds);
	  else calc_particle_bounds_periodic(w.bounds);
	}
	new_bounds = 1;
	assert(!dup_ids);
      }
      else {
	num_chunks = 1;
	chunks[0] = w.chunk;
      }
      record_time(bp_time);

      if (CLIENT_DEBUG) fprintf(stderr, "Received %"PRId64" particles and %"PRId64" fofs from id %"PRId64" (Worker %"PRId64")\n", num_p, w.num_fofs, id-NUM_READERS, chunk);
      if (new_bounds && TEMPORAL_HALO_FINDING) {
	new_bounds = 0;
	if (!memcmp(w.bounds, zero_bounds, sizeof(float)*6))
	  load_previous_halos(snap, w.chunk, NULL); //Single processor
	else load_previous_halos(snap, w.chunk, w.bounds);
      }

      record_time(ph_time);
      do_workunit(&w, fofs);
      record_time(work_time);

      if (CLIENT_DEBUG) fprintf(stderr, "Analyzed %"PRId64" particles and %"PRId64" fofs from id %"PRId64", and found %"PRId64" halos. (Worker %"PRId64")\n", w.num_particles, w.num_fofs, id-NUM_READERS, num_halos, chunk);
      w.num_halos = num_halos;
      total_pp += w.num_particles + w.num_meta_p;
      total_h += num_halos;
      total_wku++;

      //      fprintf(stderr, "Workunit done (%"PRId64" halos found; %"PRId64" chunks)!!!\n", num_halos, num_chunks);
      for (i=0; i<num_halos; i++) wrap_into_box(halos[i].pos);
      for (i=0; i<num_chunks; i++) {
	struct fof *chunk_fofs=NULL;
	struct halo *chunk_halos=NULL;
	struct extra_halo_info *chunk_ei=NULL;
	struct particle *chunk_p=NULL;
	struct workunit_info chunk_w = w;
	sort_out_halos_for_chunk(chunks[i], chunk_info[chunks[i]].bounds, &chunk_w, &chunk_fofs, &chunk_halos, &chunk_ei, &chunk_p, fofs);
	if (chunk_w.num_halos) {
	  int64_t m2 = connect_to_addr(chunk_info[chunks[i]].address,
				       chunk_info[chunks[i]].port);
	  send_to_socket_noconfirm(m2, &minus1, sizeof(int64_t));
	  send_to_socket_noconfirm(m2, "wrkd", 4);
	  send_to_socket_noconfirm(m2, &chunk_w, sizeof(struct workunit_info));
	  send_to_socket_noconfirm(m2, chunk_fofs, sizeof(struct fof)*chunk_w.num_fofs);
	  send_to_socket_noconfirm(m2, chunk_halos, sizeof(struct halo)*chunk_w.num_halos);
	  send_to_socket_noconfirm(m2, chunk_ei, sizeof(struct extra_halo_info)*chunk_w.num_halos);
	  send_to_socket(m2, chunk_p, sizeof(struct particle)*(chunk_w.num_particles+chunk_w.num_meta_p));
	  close_rsocket(m2);

	  if (chunk_p != p) {
	    free(chunk_fofs);
	    free(chunk_halos);
	    free(chunk_ei);
	    free(chunk_p);
	  }
	}
      }

      send_to_socket(m, "wrku", 4);
      record_time(send_time);
    }
    else if (!strcmp(cmd, "nmwk")) {
      if (m != s) close_rsocket(m);
      p = check_realloc(p, 0, "Freeing particles.");
      clear_prev_files();
      send_to_socket(s, "nmwk", 4);
      send_to_socket(s, &id, sizeof(int64_t));
      m = s;
    }
    else if (!strcmp(cmd, "work")) {
      assert(m==s);
      recv_from_socket(s, &id, sizeof(int64_t));
      address = recv_msg_nolength(s, address);
      port = recv_msg_nolength(s, port);
      m = connect_to_addr(address, port);
      if (m<0) {
	send_to_socket(s, "clos", 4);
	exit(1);
      }
      send_to_socket(m, &chunk, sizeof(int64_t));
      new_bounds = 1;
    }
    else if (!strcmp(cmd, "fini")) {
      record_time(f_time);
      if (profile_out) 
	fprintf(profile_out, "[Prof] S%"PRId64",C%"PRId64": %"PRId64"p,%"PRId64"h,%"PRId64"w; wt:%"PRId64"s; rcv:%"PRId64"s,%"PRId64"s; snd:%"PRId64"s; wk:%"PRId64"s; idl:%"PRId64"s\n", snap, chunk, total_pp, total_h, total_wku, idle_time, recv_time, bp_time, send_time, work_time, f_time);
      fflush(profile_out);
      exit(0);
    }
    else {
      fprintf(stderr, "[Error] Error in client protocol aw (%s)!\n", cmd);
      send_to_socket_noconfirm(s, "clos", 4);
      exit(1);
    }
  }
}

int send_workunit(int64_t sock, struct workunit_info *w,
		  struct fof **fofs, struct particle **particles,
		  int64_t **set_sizes, struct bgroup **bgroup_list,
		  int64_t *no_more_work, int64_t chunk, float *bounds) {
  if (!(*no_more_work)) 
    find_unfinished_workunit(w, fofs, particles, set_sizes, bgroup_list);
  if ((*no_more_work) || (!w->num_fofs)) {
    *no_more_work = 1;
    send_to_socket(sock, "nmwk", 4);
    return 0;
  }
  if (bounds) memcpy(w->bounds, bounds, sizeof(float)*6);
  else memset(w->bounds, 0, sizeof(float)*6);
  w->chunk = chunk;
  send_to_socket(sock, "wrku", 4);
  send_to_socket(sock, w, sizeof(struct workunit_info));
  send_to_socket(sock, *fofs, sizeof(struct fof)*w->num_fofs);
  if (w->num_particles)
    send_to_socket(sock, *particles, sizeof(struct particle)*w->num_particles);
  if (w->num_meta_fofs) {
    send_to_socket(sock, *set_sizes, sizeof(int64_t)*w->num_meta_fofs);
    send_to_socket(sock, *bgroup_list, sizeof(struct bgroup)*w->total_bg);
  }
  if ((*particles >= p) && (*particles < p+num_p)) *particles = NULL;
  return 1;
}

int64_t distribute_workloads(int64_t c, int64_t s, int64_t snap, int64_t chunk, float *bounds) {
  int64_t i, j, num_workers = 0, no_more_work = 0, workdone = 0, all_clear = 0,
    done = 0, id_offset=0, worker_chunk, hcnt, id;
  int64_t *workers = NULL, max_cs = 0, child = -1, new_w, child_has_connected=0;
  char *address = NULL, *port = NULL;
  char cmd[5] = {0};
  struct workunit_info w;
  struct particle *parts = NULL;
  struct particle *pbuffer = NULL;
  struct fof *fofs = NULL;
  struct halo *rhalos = NULL;
  struct extra_halo_info *ehi = NULL;
  int64_t *set_sizes = NULL;
  struct bgroup *bgroup_list = NULL;

  while ((num_workers || !done) && !(in_error_state)) {

    if (all_clear && !workdone && no_more_work && child_has_connected) {
      workdone = 1;
      rockstar_cleanup();
      hcnt = count_halos_to_print(bounds);
      send_to_socket_noconfirm(c, "hcnt", 4);
      send_to_socket_noconfirm(c, &hcnt, sizeof(int64_t));
      if (CLIENT_DEBUG) fprintf(stderr, "Analysis of %"PRId64" halos complete (chunk %"PRId64")\n", hcnt, chunk);
    }

    clear_rsocket_tags();
    if (!done || child > -1) tag_rsocket(c);
    if (!done) tag_rsocket(s);
    for (i=0; i<num_workers; i++) tag_rsocket(workers[i]);

    max_cs = select_rsocket(RSOCKET_READ, 0);
    for (i=0; i<max_cs; i++) {
      if (!check_rsocket_tag(i)) continue;

      if (i==s) {
	new_w = accept_connection(s,NULL,NULL);
	recv_from_socket(new_w, &worker_chunk, sizeof(int64_t));
	if (worker_chunk == chunk) {
	  assert(child < 0);
	  child = new_w;
	  child_has_connected = 1;
	}
	if (worker_chunk == -1) {
	  if (recv_from_socket(new_w, cmd, 4) <= 0) {
	    network_io_err(c);
	    for (j=0; j<num_workers; j++) close_rsocket(workers[j]);
	    num_workers = 0;
	    break;
	  }
	  if (!strcmp(cmd, "part")) {
	    int64_t num_groups;
	    recv_from_socket(new_w, &num_groups, sizeof(int64_t));
	    bgroup_list = recv_and_alloc(new_w, NULL, num_groups*sizeof(struct bgroup));
	    send_to_socket(new_w, bounds, sizeof(float)*6);
	    if (!pbuffer) pbuffer = check_realloc(NULL, sizeof(struct particle)*PARTICLE_REALLOC_NUM, "Particle buffer");
	    int64_t k = 0, l = 0;
	    struct fof tf;
	    for (j=0; j<num_groups; j++) {
	      assert(bgroup_list[j].chunk == chunk);
	      fof_of_id(bgroup_list[j].id, &tf);
	      assert(tf.num_p == bgroup_list[j].num_p);
	      for (k=0; k<tf.num_p; k++) {
		pbuffer[l] = tf.particles[k];
		l++;
		if (l==PARTICLE_REALLOC_NUM) {
		  send_to_socket(new_w, pbuffer, sizeof(struct particle)*PARTICLE_REALLOC_NUM);
		  l=0;
		}
	      }
	    }
	    if (l) send_to_socket(new_w, pbuffer, sizeof(struct particle)*l);
	    if (bgroup_list) bgroup_list = check_realloc(bgroup_list, 0, "Freeing bgroups");
	  }
	  else if (!strcmp(cmd, "wrkd")) {
	    recv_from_socket(new_w, &w, sizeof(struct workunit_info));
	    assert((w.num_particles+w.num_meta_p) >= 0);
	    fofs = recv_and_alloc(new_w, fofs, sizeof(struct fof)*w.num_fofs);
	    rhalos = recv_and_alloc(new_w, rhalos, sizeof(struct halo)*w.num_halos);
	    ehi = recv_and_alloc(new_w, ehi, sizeof(struct extra_halo_info)*w.num_halos);
	    parts = recv_and_alloc(new_w, parts, sizeof(struct particle)*(w.num_particles+w.num_meta_p));
	    integrate_finished_workunit(&w, fofs, rhalos, ehi, parts);
	  }
	  close_rsocket(new_w);
	}
	else {
	  if (send_workunit(new_w, &w, &fofs, &parts, &set_sizes, &bgroup_list,
			    &no_more_work, chunk, bounds) ||
	      (child == new_w)) {
	    workers = check_realloc(workers, sizeof(int64_t)*(num_workers+1),
				    "Allocating rockstar analysis FDs.");
	    workers[num_workers] = new_w;
	    num_workers++;
	    if (CLIENT_DEBUG) fprintf(stderr, "Got new worker (chunk %"PRId64"; wchunk %"PRId64")\n", chunk, worker_chunk);
	  } else {
	    close_rsocket(new_w);
	  }
	}
      }
      else if (i==c) {
	recv_from_socket(c, cmd, 4);
	if (!strcmp(cmd, "fini")) {
	  assert(child >= 0);
	  send_to_socket(child, "fini", 4);
	  close_connection(child, workers, &num_workers);
	  child = -1;
	  if (CLIENT_DEBUG) fprintf(stderr, "Told child to finish (chunk %"PRId64")\n", chunk);
	}
	else if (!strcmp(cmd, "work")) {
	  assert(child >= 0);
	  recv_from_socket(c, &id, sizeof(int64_t));
	  address = recv_msg_nolength(c, address);
	  port = recv_msg_nolength(c, port);
	  send_to_socket(child, "work", 4);
	  send_to_socket(child, &id, sizeof(int64_t));
	  send_msg(child, address, strlen(address)+1);
	  send_msg(child, port, strlen(port)+1);
	  if (CLIENT_DEBUG) fprintf(stderr, "Child (%"PRId64") connecting to id %"PRId64"\n", chunk, id);
	}
	else if (!strcmp(cmd, "allc")) all_clear = 1;
	else if (!strcmp(cmd, "outp")) {
	  assert(no_more_work && !done);
	  recv_from_socket(c, &id_offset, sizeof(int64_t));
	  output_halos(id_offset, snap, chunk, bounds);
	  done = 1;
	  if (CLIENT_DEBUG) fprintf(stderr, "Finished (chunk %"PRId64")\n", chunk);
	}
	else if (!strcmp(cmd, "quit")) {
	  exit(1);
	}
	else if (!strcmp(cmd, "err!")) {
	  in_error_state = 1;
	  for (j=0; j<num_workers; j++) close_rsocket(workers[j]);
	  num_workers = 0;
	  break;
	}
	else { fprintf(stderr, "[Error] Server protocol error dw (%s)!\n", cmd); exit(1); }
      }
      else {
	if (recv_from_socket(i, cmd, 4) <= 0) {
	  network_io_err(c);
	  for (j=0; j<num_workers; j++) close_rsocket(workers[j]);
	  num_workers = 0;
	  break;
	}
	if (!strcmp(cmd, "wrku")) {
	  if (!send_workunit(i, &w, &fofs, &parts, &set_sizes, &bgroup_list,
			     &no_more_work, chunk, bounds) && 
	      (child != i))
	    close_connection(i, workers, &num_workers);
	}
	else if (!strcmp(cmd, "clos")) {
	  close_connection(i, workers, &num_workers);
	}
	else if (!strcmp(cmd, "nmwk")) {
	  assert(i == child);
	  recv_from_socket(i, &id, sizeof(int64_t));
	  send_to_socket_noconfirm(c, "nmwk", 4);
	  send_to_socket_noconfirm(c, &id, sizeof(int64_t));
	}

	else { fprintf(stderr, "[Error] Client protocol error dw (%s)!\n", cmd); exit(1); }
      }
    }
  }
  check_realloc(address, 0, "Freeing address memory.");
  check_realloc(port, 0, "Freeing port memory.");
  check_realloc(workers, 0, "Freeing worker memory.");
  check_realloc(fofs, 0, "Freeing FOF memory.");
  check_realloc(parts, 0, "Freeing particle memory.");
  check_realloc(rhalos, 0, "Freeing halo memory.");
  check_realloc(ehi, 0, "Freeing extra halo info memory.");
  if (pbuffer) check_realloc(pbuffer, 0, "Freeing particle buffer");
  if (set_sizes) check_realloc(set_sizes, 0, "Freeing set sizes");
  if (bgroup_list) check_realloc(bgroup_list, 0, "Freeing bgroups");
  return id_offset;
}


void client(int64_t type) {
  int64_t snap=0, block, chunk=0, i, n, timestep, id_offset=0;
  struct binary_output_header bheader;
  uint64_t magic;
  char buffer[1024];
  char cmd[5] = {0};
  char *hostname = NULL;
  char port[10] = {0};
  float bounds[6], halo_bounds[6], box_size;
  //struct recipient *r;
  int64_t c, s = -1, portnum, readers;
  int64_t num_nodes;
  //int stat_loc = 0;

  clear_merger_tree();
  if (FORK_READERS_FROM_WRITERS) {
    num_nodes = (FORK_PROCESSORS_PER_MACHINE) ? 
      (NUM_WRITERS/FORK_PROCESSORS_PER_MACHINE) : NUM_WRITERS;
    readers = NUM_READERS/num_nodes + ((NUM_READERS%num_nodes) ? 1 : 0);
    for (i=0; i<readers; i++) {
      n = fork();
      if (n==0) { type = READER_TYPE; break; } 
      if (n<0) system_error("Couldn't fork reader process!");
    }
    if (i==readers) { type = WRITER_TYPE; }
  }

  if (FORK_PROCESSORS_PER_MACHINE && (type != READER_TYPE)) {
    for (i=1; i<FORK_PROCESSORS_PER_MACHINE; i++) {
      n = fork();
      if (n==0) break;
      if (n<0) system_error("Couldn't fork process!");
    }
  }

  srand(getpid()); /* so rand() in random_sleep() is different across procs */
  random_sleep(5*(NUM_WRITERS+NUM_READERS)/512);
  c = connect_to_addr(PARALLEL_IO_SERVER_ADDRESS, PARALLEL_IO_SERVER_PORT);
  recv_from_socket(c, &magic, sizeof(uint64_t));
  if (magic != ROCKSTAR_MAGIC) {
    fprintf(stderr, "[Error] Received invalid client responses.  Check network connectivity.\n");
    exit(1);
  }
  send_to_socket_noconfirm(c, &magic, sizeof(uint64_t));
  if (type < 0) send_to_socket_noconfirm(c, "rdwr", 4);
  if (type == READER_TYPE) send_to_socket_noconfirm(c, "read", 4);
  if (type == WRITER_TYPE) send_to_socket_noconfirm(c, "writ", 4);

  while (1) {
    recv_from_socket(c, cmd, 4);
    if (!strcmp(cmd, "err!")) network_error_cleanup();

    if (!strcmp(cmd, "read")) {
      type = READER_TYPE;
      hostname = recv_msg_nolength(c, hostname);
      send_msg(c, port, sizeof(char));
    }

    else if (!strcmp(cmd, "writ")) {
      type = WRITER_TYPE;
      hostname = recv_msg_nolength(c, hostname);
      for (i=0; i<10000 && s<0; i+=29) {
	portnum = PARALLEL_IO_WRITER_PORT+i;
	snprintf(port, 10, "%"PRId64, portnum);
	s = listen_at_addr(hostname, port);
      }
      if (i>=10000) {
	fprintf(stderr, "[Error] Couldn't start particle data server at %s:%d-%d!\n",
		hostname, (int)PARALLEL_IO_WRITER_PORT, (int)portnum);
	exit(1);
      }
      send_msg(c, port, strlen(port)+1);
      if (LIGHTCONE && strlen(LIGHTCONE_ALT_SNAPS)) {
	memcpy(LIGHTCONE_ORIGIN, LIGHTCONE_ALT_ORIGIN, sizeof(double)*3);
      }
    }

    else if (!strcmp(cmd, "snap"))
      recv_from_socket(c, &snap, sizeof(int64_t));

    else if (!strcmp(cmd, "chnk"))
      recv_from_socket(c, &chunk, sizeof(int64_t));

    else if (!strcmp(cmd, "info")) {
      if (chunk_info) { free(chunk_info[0].address); free(chunk_info); }
      chunk_info = check_realloc(NULL, sizeof(struct chunk_info)*NUM_WRITERS, "Allocating address information");
      chunk_info[0].address = recv_msg_nolength(c, NULL);
      chunk_info[0].port = chunk_info[0].address+strlen(chunk_info[0].address)+1;
      for (i=1; i<NUM_WRITERS; i++) {
	chunk_info[i].address = chunk_info[i-1].port + strlen(chunk_info[i-1].port)+1;
	chunk_info[i].port = chunk_info[i].address + strlen(chunk_info[i].address)+1;
      }
    }

    else if (!strcmp(cmd, "rdbk")) {
      assert(type == READER_TYPE);
      recv_from_socket(c, &block, sizeof(int64_t));
      if (LIGHTCONE && strlen(LIGHTCONE_ALT_SNAPS) && block >= (NUM_BLOCKS/2)) {
	if (LIGHTCONE == 1)
	  read_input_names(LIGHTCONE_ALT_SNAPS, &snapnames, &NUM_SNAPS);
	LIGHTCONE = 2;
	get_input_filename(buffer, 1024, snap, block-(NUM_BLOCKS/2));
      }
      else {
	if (LIGHTCONE == 2) {
	  LIGHTCONE = 1;
	  read_input_names(SNAPSHOT_NAMES, &snapnames, &NUM_SNAPS);
	}	  
	get_input_filename(buffer, 1024, snap, block);
      }
      read_particles(buffer);
      if (!block) output_config(NULL);
    }
      
    else if (!strcmp(cmd, "cnf?")) {
      assert(type == READER_TYPE);
      calc_particle_bounds(bounds);
      if (TRIM_OVERLAP) trim_particles(bounds);
      send_to_socket(c, "bxsz", 4);
      box_size = BOX_SIZE;
      send_to_socket(c, &box_size, sizeof(float));
      send_to_socket(c, bounds, sizeof(float)*6);
      send_to_socket(c, "cnfg", 4);
      send_config(c);
    }

    else if (!strcmp(cmd, "cnfg")) {
      assert(type == WRITER_TYPE);
      recv_config(c);
      if (LIGHTCONE) init_cosmology();
      init_time_table();
    }

    else if (!strcmp(cmd, "rdy?")) {
      send_to_socket(c, "rdy!", 4);
    }

    else if (!strcmp(cmd, "proj")) {
      assert(type == READER_TYPE);
      recv_from_socket(c, &num_proj, sizeof(int64_t));
      reset_projection_count();
      if (num_proj) {
	recv_from_socket(c, prq, sizeof(struct projection_request)*num_proj);
	if (CLIENT_DEBUG) fprintf(stderr, "Doing client projections for block %"PRId64"\n", block);
	do_projections();
	if (CLIENT_DEBUG) fprintf(stderr, "Done with client projections for block %"PRId64"\n", block);
      }
      send_to_socket(c, "cprj", 4);
      send_to_socket(c, &num_proj, sizeof(int64_t));
      for (i=0; i<num_proj; i++)
	send_to_socket(c, prj+i, sizeof(struct projection));
    }

    else if (!strcmp(cmd, "bnds")) {
      assert(type == WRITER_TYPE);
      recv_from_socket(c, bounds, sizeof(float)*6);
    }

    else if (!strcmp(cmd, "hbnd")) {
      assert(type == WRITER_TYPE);
      calc_halo_bounds(halo_bounds);
      send_to_socket(c, halo_bounds, sizeof(float)*6);
    }

    else if (!strcmp(cmd, "rcpt")) {
      assert(type == READER_TYPE);
      add_recipient(sizeof(struct particle), c);
    }

    else if (!strcmp(cmd, "xfrp")) {
      if (type == READER_TYPE) send_particles(c, bounds);
      if (type == WRITER_TYPE) transfer_stuff(s,c,0);
      if (in_error_state) network_error_cleanup();
    }

    else if (!strcmp(cmd, "fofs")) {
      assert(type == WRITER_TYPE);
      recv_from_socket(c, &chunk, sizeof(int64_t));
      if (EXTRA_PROFILING && !profile_out) {
	snprintf(buffer, 1024, "%s/profiling", OUTBASE); 
	mkdir(buffer, 0777);
	snprintf(buffer, 1024, "%s/profiling/profile.%"PRId64, OUTBASE, chunk); 
	profile_out = check_fopen(buffer, "w"); //Truncate
	fclose(profile_out);
	profile_out = check_fopen(buffer, "a");
      }
      int64_t time_start = time(NULL);
      rockstar(bounds, 1);
      int64_t time_middle = time(NULL);
      set_bp_chunk(chunk);
      send_to_socket(c, "done", 4);
      int64_t time_end = time(NULL);
      if (CLIENT_DEBUG) fprintf(stderr, "Found %"PRId64" fofs in chunk %"PRId64"\n", num_all_fofs, chunk);
      if (profile_out) {
	fprintf(profile_out, "[Prof] S%"PRId64",C%"PRId64" %"PRId64"s: %"PRId64" fofs, %"PRId64" particles, %"PRId64"s for conf.\n", snap, chunk, (time_middle-time_start), num_all_fofs, num_p, (time_end-time_middle));
	fflush(profile_out);
      }
    }

    else if (!strcmp(cmd, "rock")) {
      n = fork();
      if (!n) accept_workloads(hostname, port, snap, chunk);
      if (n<0) system_error("Couldn't fork halo analysis process!");
      id_offset = distribute_workloads(c, s, snap, chunk, bounds);
      check_waitpid(n);
      if (in_error_state) network_error_cleanup();
      send_to_socket(c, "done", 4);
    }

    else if (!strcmp(cmd, "outp")) {
      assert(type == WRITER_TYPE);
      id_offset = 0; //Only used for dumping particles
      free_halos();
      p = check_realloc(p, 0, "Freeing particle memory");
      num_p = 0;
      send_to_socket(c, "done", 4);
    }

    else if (!strcmp(cmd, "gbds")) {
      load_binary_header(snap, chunk, &bheader);
      send_to_socket(c, &(bheader.bounds), sizeof(float)*6);
    }

    else if (!strcmp(cmd, "pbds")) {
      if (!p_bounds) 
	check_realloc_s(p_bounds, sizeof(struct prev_bounds), NUM_WRITERS);
      recv_from_socket(c, p_bounds, sizeof(struct prev_bounds)*NUM_WRITERS);
      prev_snap = snap;
    }

    else if (!strcmp(cmd, "rcph")) {
      assert(type == WRITER_TYPE);
      add_recipient(sizeof(struct halo), c);
    }

    else if (!strcmp(cmd, "xfrh")) {
      assert(type == WRITER_TYPE);
      recv_from_socket(c, &timestep, sizeof(int64_t));
      n = fork();
      if (!n) send_halos(hostname, port, snap, chunk);
      if (n<0) system_error("Couldn't fork merger tree process!");
      transfer_stuff(s,c,timestep);
      check_waitpid(n);
      if (in_error_state) {
	network_error_cleanup();
	continue;
      }
      if (timestep == 1) init_descendants();
      else if (timestep == 2) connect_particle_ids_to_halo_ids();
      clear_recipients();
    }

    else if (!strcmp(cmd, "rcpb")) {
      assert(type == WRITER_TYPE);
      add_recipient(sizeof(struct bparticle), c);
    }

    else if (!strcmp(cmd, "xfrb")) {
      assert(type == WRITER_TYPE);
      num_new_bp = 0;
      n = fork();
      if (!n) send_bparticles(hostname, port);
      if (n<0) system_error("Couldn't fork boundary particle process!");
      transfer_stuff(s,c,0);
      check_waitpid(n);
      if (in_error_state) {
	network_error_cleanup();
	continue;
      }
      clear_recipients();
      build_bgroup_links();
      clear_bp_data();
    }

    else if (!strcmp(cmd, "lnkb")) {
      assert(type == WRITER_TYPE);
      n = fork();
      if (!n) collect_bgroups(chunk);
      if (n<0) system_error("Couldn't fork boundary group process!");
      transfer_stuff(s,c,0);
      check_waitpid(n);
      if (in_error_state) {
	network_error_cleanup();
	continue;
      }
      convert_bgroups_to_metafofs();
      clear_bg_data();
    }

    else if (!strcmp(cmd, "rchp")) {
      assert(type == WRITER_TYPE);
      add_recipient(sizeof(struct sphere_request), c);
    }

    else if (!strcmp(cmd, "bgc2")) {
      assert(type == WRITER_TYPE);
      init_extended_particle_tree();
      n = fork();
      if (!n) gather_spheres(hostname, port, bounds, id_offset, snap, chunk);
      if (n<0) system_error("Couldn't fork boundary group process!");
      transfer_stuff(s,c,0);
      check_waitpid(n);
      if (in_error_state) {
	network_error_cleanup();
	continue;
      }
      free_extended_particle_tree();
      clear_recipients();
    }

    else if (!strcmp(cmd, "merg")) {
      assert(type == WRITER_TYPE);
      calculate_descendants();
    }

    else if (!strcmp(cmd, "genc")) {
      assert(type == WRITER_TYPE);
      char *cat = NULL;
      int64_t cat_length, head_length, location;
      cat = gen_merger_catalog(snap, chunk, halos1, head1.num_halos, 
			       &cat_length, &head_length);
      send_to_socket_noconfirm(c, &head_length, sizeof(int64_t));
      send_to_socket_noconfirm(c, &cat_length, sizeof(int64_t));
      recv_from_socket(c, cmd, 4);
      assert(!strcmp(cmd, "outc"));
      recv_from_socket(c, &location, sizeof(int64_t));
      output_merger_catalog(snap, chunk, location, cat_length, cat);
      clear_merger_tree();
    }

    else if (!strcmp(cmd, "delb")) {
      assert(type == WRITER_TYPE);
      if (DELETE_BINARY_OUTPUT_AFTER_FINISHED)
	delete_binary(snap, chunk);
    }

    else if (!strcmp(cmd, "free")) {
      free_halos();
      particle_cleanup();
      clear_final_bg_data();
      check_mtrim();
    }

    else if (!strcmp(cmd, "rpos")) {
      assert(type == WRITER_TYPE);
      snprintf(buffer, 1024, "%s %"PRId64" %"PRId64" %"PRId64" %"PRId64" '%s'",
	      RUN_PARALLEL_ON_SUCCESS, snap, NUM_SNAPS, chunk, NUM_WRITERS,
	      ROCKSTAR_CONFIG_FILENAME);
      if (system(buffer) != 0) {
	system_error("Running external parallel analysis process failed.");
	send_to_socket(c, "fail", 4);
	return;
      }
      send_to_socket(c, "done", 4);
    }
    
    else if (!strcmp(cmd, "quit")) {
      close_rsocket(c);
      if (hostname) free(hostname);
      return;
    }
  }
}
