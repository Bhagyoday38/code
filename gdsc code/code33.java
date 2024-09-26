import java.util.*;

public class ContactDictionary {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        
        Map<String, String> contacts = new HashMap<>();

        
        System.out.println("Enter the name of the contact: ");
        String name = scanner.nextLine();
        System.out.println("Enter the email address of the contact: ");
        String email = scanner.nextLine();
        System.out.println("Enter the phone number of the contact: ");
        String phone = scanner.nextLine();
        contacts.put(name, email + ", " + phone);

        
        System.out.println("Enter the name of the contact you want to update: ");
        name = scanner.nextLine();
        if (contacts.containsKey(name)) {
            System.out.println("Enter the new email address of the contact: ");
            email = scanner.nextLine();
            System.out.println("Enter the new phone number of the contact: ");
            phone = scanner.nextLine();
            contacts.put(name, email + ", " + phone);
        } else {
            System.out.println("Contact not found.");
        }

        
        System.out.println("Enter the name of the contact you want to delete: ");
        name = scanner.nextLine();
        if (contacts.containsKey(name)) {
            contacts.remove(name);
        } else {
            System.out.println("Contact not found.");
        }

        
        System.out.println("Contacts:");
        for (Map.Entry<String, String> entry : contacts.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }
}
