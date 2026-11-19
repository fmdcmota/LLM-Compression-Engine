#include <iostream>

#include <unistd.h>
#include <sys/wait.h>

// for example, let's "ls"
int run(const char *dir) {
   int pid, status;
   // first we fork the process
   if (pid = fork()) {
       // pid != 0: this is the parent process (i.e. our process)
       waitpid(pid, &status, 0); // wait for the child to exit
   } else {
       /* pid == 0: this is the child process. now let's load the
          "ls" program into this process and run it */
       // load it. there are more exec__ functions, try 'man 3 exec'
       // execl takes the arguments as parameters. execv takes them as an array
       // this is execl though, so:
       //      exec         argv[0]  argv[1] end
      execl("./llama.cpp/build/bin", "./llama.cpp/build/bin", "-m", "/mnt/c/Users/nanoc/PROGRAMMING/LLM-Compression-Engine/models/tinyllama-15M-stories-Q3_K_M.gguf", (char *)NULL);
       /* exec does not return unless the program couldn't be started. 
          when the child process stops, the waitpid() above will return.
       */


   }
   return status; // this is the parent process again.
}


int main() {

   std::cout << "ls'ing /" << std::endl;
   std::cout << "returned: " << run("/mnt/c/Users/nanoc/PROGRAMMING/LLM-Compression-Engine/llama.cpp$") << std::endl;

   return 0;
}


